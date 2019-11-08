module LibLightGBM

import Libdl
import Base.Sys: islinux, isapple, iswindows

# TODO refactor into the standard deps format
# https://github.com/JuliaOpt/CSDP.jl/tree/master/deps

function check_deps()
    if isapple()
        if !isfile("/usr/local/Cellar/lightgbm/2.3.0/lib/lib_lightgbm.dylib")
            error("can't find shared lib")
        end
    elseif islinux()
        error("linux not supported")
    elseif iswindows()
        error("windows not supported")
    end
end

# Module initialization function
function __init__()
    check_deps()
end



using CEnum

include(joinpath(@__DIR__, "wrapper", "ctypes.jl"))
export Ctm, Ctime_t, Cclock_t

include(joinpath(@__DIR__, "wrapper", "liblightgbm_common.jl"))
include(joinpath(@__DIR__, "wrapper", "liblightgbm_api.jl"))

# export everything
foreach(names(@__MODULE__, all=true)) do s
   if startswith(string(s), "LGBM_") || startswith(string(s), "C_API_")
       @eval export $s
   end
end

export DatasetHandler, BoosterHandle

end # module
