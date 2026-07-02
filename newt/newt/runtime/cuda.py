"""Minimal ctypes bindings to the CUDA driver API and NVRTC.

This is the entire "backend" of newt: we hand NVRTC a CUDA C++ source string,
get back a cubin, load it with the driver API, and launch it on torch's
current stream. No dependency on the CUDA runtime, cuda-python, or pynvml.
"""

import ctypes
import ctypes.util
import functools
import glob
import os
import sys

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------


def _candidate_dirs():
    dirs = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        dirs += [os.path.join(cuda_path, "bin"), os.path.join(cuda_path, "bin", "x64")]
    if sys.platform == "win32":
        dirs += glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin")
        dirs += glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\x64")
    try:
        import torch

        dirs.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
    except ImportError:
        pass
    try:
        import nvidia.cuda_nvrtc

        dirs.append(os.path.join(os.path.dirname(nvidia.cuda_nvrtc.__file__), "bin"))
        dirs.append(os.path.join(os.path.dirname(nvidia.cuda_nvrtc.__file__), "lib"))
    except ImportError:
        pass
    return dirs


@functools.cache
def _load_nvrtc():
    if sys.platform == "win32":
        names = []
        for d in _candidate_dirs():
            names += sorted(glob.glob(os.path.join(d, "nvrtc64_*.dll")), reverse=True)
        names = [n for n in names if "builtins" not in os.path.basename(n)]
    else:
        names = ["libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12"]
    errors = []
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError as e:
            errors.append(f"{name}: {e}")
    raise RuntimeError(
        "newt: could not load NVRTC. Install the CUDA toolkit or set CUDA_PATH.\n"
        + "\n".join(errors[:5])
    )


@functools.cache
def _load_driver():
    name = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
    try:
        return ctypes.CDLL(name)
    except OSError as e:
        raise RuntimeError(f"newt: could not load CUDA driver library {name}: {e}")


@functools.cache
def cuda_include_dir():
    """Directory containing cuda_fp16.h etc., passed to NVRTC with -I."""
    candidates = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(os.path.join(cuda_path, "include"))
    if sys.platform == "win32":
        candidates += sorted(
            glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\include"),
            reverse=True,
        )
    else:
        candidates += ["/usr/local/cuda/include"]
    try:
        import nvidia.cuda_runtime

        candidates.append(os.path.join(os.path.dirname(nvidia.cuda_runtime.__file__), "include"))
    except ImportError:
        pass
    for c in candidates:
        if os.path.isfile(os.path.join(c, "cuda_fp16.h")):
            return c
    return None


# ---------------------------------------------------------------------------
# NVRTC
# ---------------------------------------------------------------------------


class NVRTCError(RuntimeError):
    pass


def _nvrtc_check(nvrtc, code, prog=None):
    if code == 0:
        return
    nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p
    msg = nvrtc.nvrtcGetErrorString(code).decode()
    log = ""
    if prog is not None:
        size = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
        buf = ctypes.create_string_buffer(size.value)
        nvrtc.nvrtcGetProgramLog(prog, buf)
        log = buf.value.decode(errors="replace")
    raise NVRTCError(f"NVRTC error: {msg}\n{log}")


def compile_cuda(source: str, name: str, arch: str, extra_opts=()) -> bytes:
    """Compile CUDA C++ source to a cubin for the given arch (e.g. 'sm_120')."""
    nvrtc = _load_nvrtc()
    prog = ctypes.c_void_p()
    _nvrtc_check(
        nvrtc,
        nvrtc.nvrtcCreateProgram(
            ctypes.byref(prog),
            source.encode(),
            f"{name}.cu".encode(),
            0,
            None,
            None,
        ),
    )
    try:
        opts = [f"--gpu-architecture={arch}", "--std=c++17", "-DNDEBUG"]
        inc = cuda_include_dir()
        if inc:
            opts.append(f"-I{inc}")
        opts += list(extra_opts)
        c_opts = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
        code = nvrtc.nvrtcCompileProgram(prog, len(opts), c_opts)
        _nvrtc_check(nvrtc, code, prog)
        size = ctypes.c_size_t()
        _nvrtc_check(nvrtc, nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(size)), prog)
        buf = ctypes.create_string_buffer(size.value)
        _nvrtc_check(nvrtc, nvrtc.nvrtcGetCUBIN(prog, buf), prog)
        return buf.raw
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))


# ---------------------------------------------------------------------------
# CUDA driver API
# ---------------------------------------------------------------------------

CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1


class CUDAError(RuntimeError):
    pass


def _cu_check(cu, code):
    if code == 0:
        return
    s = ctypes.c_char_p()
    cu.cuGetErrorString(code, ctypes.byref(s))
    raise CUDAError(f"CUDA driver error {code}: {(s.value or b'?').decode()}")


@functools.cache
def _init_context():
    """Initialize the driver and make the primary context current.

    torch also uses the primary context, so modules/streams interoperate.
    """
    cu = _load_driver()
    _cu_check(cu, cu.cuInit(0))
    ctx = ctypes.c_void_p()
    _cu_check(cu, cu.cuCtxGetCurrent(ctypes.byref(ctx)))
    if not ctx.value:
        dev = ctypes.c_int()
        _cu_check(cu, cu.cuDeviceGet(ctypes.byref(dev), 0))
        _cu_check(cu, cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev))
        _cu_check(cu, cu.cuCtxSetCurrent(ctx))
    return cu


class Kernel:
    """A loaded GPU function, launchable with raw ctypes arguments."""

    def __init__(self, cubin: bytes, func_name: str, shared_bytes: int = 0):
        cu = _init_context()
        self._cu = cu
        self.module = ctypes.c_void_p()
        _cu_check(cu, cu.cuModuleLoadData(ctypes.byref(self.module), cubin))
        self.func = ctypes.c_void_p()
        _cu_check(
            cu, cu.cuModuleGetFunction(ctypes.byref(self.func), self.module, func_name.encode())
        )
        if shared_bytes > 48 * 1024:
            _cu_check(
                cu,
                cu.cuFuncSetAttribute(
                    self.func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_bytes
                ),
            )
        self.shared_bytes = shared_bytes

    def launch(self, grid, block, args, stream=0):
        """args: list of ctypes objects (c_void_p / c_int32 / c_float / ...)."""
        ptrs = (ctypes.c_void_p * len(args))(
            *[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args]
        )
        gx, gy, gz = (list(grid) + [1, 1, 1])[:3]
        bx, by, bz = (list(block) + [1, 1, 1])[:3]
        _cu_check(
            self._cu,
            self._cu.cuLaunchKernel(
                self.func,
                gx, gy, gz,
                bx, by, bz,
                self.shared_bytes,
                ctypes.c_void_p(stream),
                ptrs,
                None,
            ),
        )
