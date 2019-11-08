using Clang

LIBCLANG_HEADERS = [ joinpath(@__DIR__, "c_api_2.3.0.h") ];
LIBCLANG_INCLUDE = ""
LIBNAME = "liblightgbm"

wc = init(; headers = LIBCLANG_HEADERS,
            output_file = joinpath(@__DIR__,  "$(LIBNAME)_api.jl"),
            common_file = joinpath(@__DIR__,  "$(LIBNAME)_common.jl"),
            clang_includes = vcat(LIBCLANG_INCLUDE, CLANG_INCLUDE),
            clang_args = ["-I", joinpath(LIBCLANG_INCLUDE, "..")],
            header_wrapped = (root, current)->root == current,
            header_library = x-> LIBNAME,
            clang_diagnostics = true,
            )

run(wc)