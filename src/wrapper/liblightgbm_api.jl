# Julia wrapper for header: c_api_2.3.0.h
# Automatically generated using Clang.jl

# TODO define this in another file not this one!
const liblightgbm = "/usr/local/Cellar/lightgbm/2.3.0/lib/lib_lightgbm.dylib"

function LGBM_GetLastError()
    ccall((:LGBM_GetLastError, liblightgbm), Cstring, ())
end

function LGBM_DatasetCreateFromFile(filename, parameters, reference, out)
    ccall((:LGBM_DatasetCreateFromFile, liblightgbm), Cint, (Cstring, Cstring, DatasetHandle, Ptr{DatasetHandle}), filename, parameters, reference, out)
end

function LGBM_DatasetCreateFromSampledColumn(sample_data, sample_indices, ncol, num_per_col, num_sample_row, num_total_row, parameters, out)
    ccall((:LGBM_DatasetCreateFromSampledColumn, liblightgbm), Cint, (Ptr{Ptr{Cdouble}}, Ptr{Ptr{Cint}}, Cint, Ptr{Cint}, Cint, Cint, Cstring, Ptr{DatasetHandle}), sample_data, sample_indices, ncol, num_per_col, num_sample_row, num_total_row, parameters, out)
end

function LGBM_DatasetCreateByReference(reference, num_total_row, out)
    ccall((:LGBM_DatasetCreateByReference, liblightgbm), Cint, (DatasetHandle, Cint, Ptr{DatasetHandle}), reference, num_total_row, out)
end

function LGBM_DatasetPushRows(dataset, data, data_type, nrow, ncol, start_row)
    ccall((:LGBM_DatasetPushRows, liblightgbm), Cint, (DatasetHandle, Ptr{Cvoid}, Cint, Cint, Cint, Cint), dataset, data, data_type, nrow, ncol, start_row)
end

function LGBM_DatasetPushRowsByCSR(dataset, indptr, indptr_type, indices, data, data_type, nindptr, nelem, num_col, start_row)
    ccall((:LGBM_DatasetPushRowsByCSR, liblightgbm), Cint, (DatasetHandle, Ptr{Cvoid}, Cint, Ptr{Cint}, Ptr{Cvoid}, Cint, Cint, Cint, Cint, Cint), dataset, indptr, indptr_type, indices, data, data_type, nindptr, nelem, num_col, start_row)
end

function LGBM_DatasetCreateFromCSR(indptr, indptr_type, indices, data, data_type, nindptr, nelem, num_col, parameters, reference, out)
    ccall((:LGBM_DatasetCreateFromCSR, liblightgbm), Cint, (Ptr{Cvoid}, Cint, Ptr{Cint}, Ptr{Cvoid}, Cint, Cint, Cint, Cint, Cstring, DatasetHandle, Ptr{DatasetHandle}), indptr, indptr_type, indices, data, data_type, nindptr, nelem, num_col, parameters, reference, out)
end

function LGBM_DatasetCreateFromCSRFunc(get_row_funptr, num_rows, num_col, parameters, reference, out)
    ccall((:LGBM_DatasetCreateFromCSRFunc, liblightgbm), Cint, (Ptr{Cvoid}, Cint, Cint, Cstring, DatasetHandle, Ptr{DatasetHandle}), get_row_funptr, num_rows, num_col, parameters, reference, out)
end

function LGBM_DatasetCreateFromCSC(col_ptr, col_ptr_type, indices, data, data_type, ncol_ptr, nelem, num_row, parameters, reference, out)
    ccall((:LGBM_DatasetCreateFromCSC, liblightgbm), Cint, (Ptr{Cvoid}, Cint, Ptr{Cint}, Ptr{Cvoid}, Cint, Cint, Cint, Cint, Cstring, DatasetHandle, Ptr{DatasetHandle}), col_ptr, col_ptr_type, indices, data, data_type, ncol_ptr, nelem, num_row, parameters, reference, out)
end

function LGBM_DatasetCreateFromMat(data, data_type, nrow, ncol, is_row_major, parameters, reference, out)
    ccall((:LGBM_DatasetCreateFromMat, liblightgbm), Cint, (Ptr{Cvoid}, Cint, Cint, Cint, Cint, Cstring, DatasetHandle, Ptr{DatasetHandle}), data, data_type, nrow, ncol, is_row_major, parameters, reference, out)
end

function LGBM_DatasetCreateFromMats(nmat, data, data_type, nrow, ncol, is_row_major, parameters, reference, out)
    ccall((:LGBM_DatasetCreateFromMats, liblightgbm), Cint, (Cint, Ptr{Ptr{Cvoid}}, Cint, Ptr{Cint}, Cint, Cint, Cstring, DatasetHandle, Ptr{DatasetHandle}), nmat, data, data_type, nrow, ncol, is_row_major, parameters, reference, out)
end

function LGBM_DatasetGetSubset(handle, used_row_indices, num_used_row_indices, parameters, out)
    ccall((:LGBM_DatasetGetSubset, liblightgbm), Cint, (DatasetHandle, Ptr{Cint}, Cint, Cstring, Ptr{DatasetHandle}), handle, used_row_indices, num_used_row_indices, parameters, out)
end

function LGBM_DatasetSetFeatureNames(handle, feature_names, num_feature_names)
    ccall((:LGBM_DatasetSetFeatureNames, liblightgbm), Cint, (DatasetHandle, Ptr{Cstring}, Cint), handle, feature_names, num_feature_names)
end

function LGBM_DatasetGetFeatureNames(handle, feature_names, num_feature_names)
    ccall((:LGBM_DatasetGetFeatureNames, liblightgbm), Cint, (DatasetHandle, Ptr{Cstring}, Ptr{Cint}), handle, feature_names, num_feature_names)
end

function LGBM_DatasetFree(handle)
    ccall((:LGBM_DatasetFree, liblightgbm), Cint, (DatasetHandle,), handle)
end

function LGBM_DatasetSaveBinary(handle, filename)
    ccall((:LGBM_DatasetSaveBinary, liblightgbm), Cint, (DatasetHandle, Cstring), handle, filename)
end

function LGBM_DatasetDumpText(handle, filename)
    ccall((:LGBM_DatasetDumpText, liblightgbm), Cint, (DatasetHandle, Cstring), handle, filename)
end

function LGBM_DatasetSetField(handle, field_name, field_data, num_element, type)
    ccall((:LGBM_DatasetSetField, liblightgbm), Cint, (DatasetHandle, Cstring, Ptr{Cvoid}, Cint, Cint), handle, field_name, field_data, num_element, type)
end

function LGBM_DatasetGetField(handle, field_name, out_len, out_ptr, out_type)
    ccall((:LGBM_DatasetGetField, liblightgbm), Cint, (DatasetHandle, Cstring, Ptr{Cint}, Ptr{Ptr{Cvoid}}, Ptr{Cint}), handle, field_name, out_len, out_ptr, out_type)
end

function LGBM_DatasetUpdateParam(handle, parameters)
    ccall((:LGBM_DatasetUpdateParam, liblightgbm), Cint, (DatasetHandle, Cstring), handle, parameters)
end

function LGBM_DatasetGetNumData(handle, out)
    ccall((:LGBM_DatasetGetNumData, liblightgbm), Cint, (DatasetHandle, Ptr{Cint}), handle, out)
end

function LGBM_DatasetGetNumFeature(handle, out)
    ccall((:LGBM_DatasetGetNumFeature, liblightgbm), Cint, (DatasetHandle, Ptr{Cint}), handle, out)
end

function LGBM_DatasetAddFeaturesFrom(target, source)
    ccall((:LGBM_DatasetAddFeaturesFrom, liblightgbm), Cint, (DatasetHandle, DatasetHandle), target, source)
end

function LGBM_BoosterCreate(train_data, parameters, out)
    ccall((:LGBM_BoosterCreate, liblightgbm), Cint, (DatasetHandle, Cstring, Ptr{BoosterHandle}), train_data, parameters, out)
end

function LGBM_BoosterCreateFromModelfile(filename, out_num_iterations, out)
    ccall((:LGBM_BoosterCreateFromModelfile, liblightgbm), Cint, (Cstring, Ptr{Cint}, Ptr{BoosterHandle}), filename, out_num_iterations, out)
end

function LGBM_BoosterLoadModelFromString(model_str, out_num_iterations, out)
    ccall((:LGBM_BoosterLoadModelFromString, liblightgbm), Cint, (Cstring, Ptr{Cint}, Ptr{BoosterHandle}), model_str, out_num_iterations, out)
end

function LGBM_BoosterFree(handle)
    ccall((:LGBM_BoosterFree, liblightgbm), Cint, (BoosterHandle,), handle)
end

function LGBM_BoosterShuffleModels(handle, start_iter, end_iter)
    ccall((:LGBM_BoosterShuffleModels, liblightgbm), Cint, (BoosterHandle, Cint, Cint), handle, start_iter, end_iter)
end

function LGBM_BoosterMerge(handle, other_handle)
    ccall((:LGBM_BoosterMerge, liblightgbm), Cint, (BoosterHandle, BoosterHandle), handle, other_handle)
end

function LGBM_BoosterAddValidData(handle, valid_data)
    ccall((:LGBM_BoosterAddValidData, liblightgbm), Cint, (BoosterHandle, DatasetHandle), handle, valid_data)
end

function LGBM_BoosterResetTrainingData(handle, train_data)
    ccall((:LGBM_BoosterResetTrainingData, liblightgbm), Cint, (BoosterHandle, DatasetHandle), handle, train_data)
end

function LGBM_BoosterResetParameter(handle, parameters)
    ccall((:LGBM_BoosterResetParameter, liblightgbm), Cint, (BoosterHandle, Cstring), handle, parameters)
end

function LGBM_BoosterGetNumClasses(handle, out_len)
    ccall((:LGBM_BoosterGetNumClasses, liblightgbm), Cint, (BoosterHandle, Ptr{Cint}), handle, out_len)
end

function LGBM_BoosterUpdateOneIter(handle, is_finished)
    ccall((:LGBM_BoosterUpdateOneIter, liblightgbm), Cint, (BoosterHandle, Ptr{Cint}), handle, is_finished)
end

function LGBM_BoosterRefit(handle, leaf_preds, nrow, ncol)
    ccall((:LGBM_BoosterRefit, liblightgbm), Cint, (BoosterHandle, Ptr{Cint}, Cint, Cint), handle, leaf_preds, nrow, ncol)
end

function LGBM_BoosterUpdateOneIterCustom(handle, grad, hess, is_finished)
    ccall((:LGBM_BoosterUpdateOneIterCustom, liblightgbm), Cint, (BoosterHandle, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cint}), handle, grad, hess, is_finished)
end

function LGBM_BoosterRollbackOneIter(handle)
    ccall((:LGBM_BoosterRollbackOneIter, liblightgbm), Cint, (BoosterHandle,), handle)
end

function LGBM_BoosterGetCurrentIteration(handle, out_iteration)
    ccall((:LGBM_BoosterGetCurrentIteration, liblightgbm), Cint, (BoosterHandle, Ptr{Cint}), handle, out_iteration)
end

function LGBM_BoosterNumModelPerIteration(handle, out_tree_per_iteration)
    ccall((:LGBM_BoosterNumModelPerIteration, liblightgbm), Cint, (BoosterHandle, Ptr{Cint}), handle, out_tree_per_iteration)
end

function LGBM_BoosterNumberOfTotalModel(handle, out_models)
    ccall((:LGBM_BoosterNumberOfTotalModel, liblightgbm), Cint, (BoosterHandle, Ptr{Cint}), handle, out_models)
end

function LGBM_BoosterGetEvalCounts(handle, out_len)
    ccall((:LGBM_BoosterGetEvalCounts, liblightgbm), Cint, (BoosterHandle, Ptr{Cint}), handle, out_len)
end

function LGBM_BoosterGetEvalNames(handle, out_len, out_strs)
    ccall((:LGBM_BoosterGetEvalNames, liblightgbm), Cint, (BoosterHandle, Ptr{Cint}, Ptr{Cstring}), handle, out_len, out_strs)
end

function LGBM_BoosterGetFeatureNames(handle, out_len, out_strs)
    ccall((:LGBM_BoosterGetFeatureNames, liblightgbm), Cint, (BoosterHandle, Ptr{Cint}, Ptr{Cstring}), handle, out_len, out_strs)
end

function LGBM_BoosterGetNumFeature(handle, out_len)
    ccall((:LGBM_BoosterGetNumFeature, liblightgbm), Cint, (BoosterHandle, Ptr{Cint}), handle, out_len)
end

function LGBM_BoosterGetEval(handle, data_idx, out_len, out_results)
    ccall((:LGBM_BoosterGetEval, liblightgbm), Cint, (BoosterHandle, Cint, Ptr{Cint}, Ptr{Cdouble}), handle, data_idx, out_len, out_results)
end

function LGBM_BoosterGetNumPredict(handle, data_idx, out_len)
    ccall((:LGBM_BoosterGetNumPredict, liblightgbm), Cint, (BoosterHandle, Cint, Ptr{Cint}), handle, data_idx, out_len)
end

function LGBM_BoosterGetPredict(handle, data_idx, out_len, out_result)
    ccall((:LGBM_BoosterGetPredict, liblightgbm), Cint, (BoosterHandle, Cint, Ptr{Cint}, Ptr{Cdouble}), handle, data_idx, out_len, out_result)
end

function LGBM_BoosterPredictForFile(handle, data_filename, data_has_header, predict_type, num_iteration, parameter, result_filename)
    ccall((:LGBM_BoosterPredictForFile, liblightgbm), Cint, (BoosterHandle, Cstring, Cint, Cint, Cint, Cstring, Cstring), handle, data_filename, data_has_header, predict_type, num_iteration, parameter, result_filename)
end

function LGBM_BoosterCalcNumPredict(handle, num_row, predict_type, num_iteration, out_len)
    ccall((:LGBM_BoosterCalcNumPredict, liblightgbm), Cint, (BoosterHandle, Cint, Cint, Cint, Ptr{Cint}), handle, num_row, predict_type, num_iteration, out_len)
end

function LGBM_BoosterPredictForCSR(handle, indptr, indptr_type, indices, data, data_type, nindptr, nelem, num_col, predict_type, num_iteration, parameter, out_len, out_result)
    ccall((:LGBM_BoosterPredictForCSR, liblightgbm), Cint, (BoosterHandle, Ptr{Cvoid}, Cint, Ptr{Cint}, Ptr{Cvoid}, Cint, Cint, Cint, Cint, Cint, Cint, Cstring, Ptr{Cint}, Ptr{Cdouble}), handle, indptr, indptr_type, indices, data, data_type, nindptr, nelem, num_col, predict_type, num_iteration, parameter, out_len, out_result)
end

function LGBM_BoosterPredictForCSRSingleRow(handle, indptr, indptr_type, indices, data, data_type, nindptr, nelem, num_col, predict_type, num_iteration, parameter, out_len, out_result)
    ccall((:LGBM_BoosterPredictForCSRSingleRow, liblightgbm), Cint, (BoosterHandle, Ptr{Cvoid}, Cint, Ptr{Cint}, Ptr{Cvoid}, Cint, Cint, Cint, Cint, Cint, Cint, Cstring, Ptr{Cint}, Ptr{Cdouble}), handle, indptr, indptr_type, indices, data, data_type, nindptr, nelem, num_col, predict_type, num_iteration, parameter, out_len, out_result)
end

function LGBM_BoosterPredictForCSC(handle, col_ptr, col_ptr_type, indices, data, data_type, ncol_ptr, nelem, num_row, predict_type, num_iteration, parameter, out_len, out_result)
    ccall((:LGBM_BoosterPredictForCSC, liblightgbm), Cint, (BoosterHandle, Ptr{Cvoid}, Cint, Ptr{Cint}, Ptr{Cvoid}, Cint, Cint, Cint, Cint, Cint, Cint, Cstring, Ptr{Cint}, Ptr{Cdouble}), handle, col_ptr, col_ptr_type, indices, data, data_type, ncol_ptr, nelem, num_row, predict_type, num_iteration, parameter, out_len, out_result)
end

function LGBM_BoosterPredictForMat(handle, data, data_type, nrow, ncol, is_row_major, predict_type, num_iteration, parameter, out_len, out_result)
    ccall((:LGBM_BoosterPredictForMat, liblightgbm), Cint, (BoosterHandle, Ptr{Cvoid}, Cint, Cint, Cint, Cint, Cint, Cint, Cstring, Ptr{Cint}, Ptr{Cdouble}), handle, data, data_type, nrow, ncol, is_row_major, predict_type, num_iteration, parameter, out_len, out_result)
end

function LGBM_BoosterPredictForMatSingleRow(handle, data, data_type, ncol, is_row_major, predict_type, num_iteration, parameter, out_len, out_result)
    ccall((:LGBM_BoosterPredictForMatSingleRow, liblightgbm), Cint, (BoosterHandle, Ptr{Cvoid}, Cint, Cint, Cint, Cint, Cint, Cstring, Ptr{Cint}, Ptr{Cdouble}), handle, data, data_type, ncol, is_row_major, predict_type, num_iteration, parameter, out_len, out_result)
end

function LGBM_BoosterPredictForMats(handle, data, data_type, nrow, ncol, predict_type, num_iteration, parameter, out_len, out_result)
    ccall((:LGBM_BoosterPredictForMats, liblightgbm), Cint, (BoosterHandle, Ptr{Ptr{Cvoid}}, Cint, Cint, Cint, Cint, Cint, Cstring, Ptr{Cint}, Ptr{Cdouble}), handle, data, data_type, nrow, ncol, predict_type, num_iteration, parameter, out_len, out_result)
end

function LGBM_BoosterSaveModel(handle, start_iteration, num_iteration, filename)
    ccall((:LGBM_BoosterSaveModel, liblightgbm), Cint, (BoosterHandle, Cint, Cint, Cstring), handle, start_iteration, num_iteration, filename)
end

function LGBM_BoosterSaveModelToString(handle, start_iteration, num_iteration, buffer_len, out_len, out_str)
    ccall((:LGBM_BoosterSaveModelToString, liblightgbm), Cint, (BoosterHandle, Cint, Cint, Cint, Ptr{Cint}, Cstring), handle, start_iteration, num_iteration, buffer_len, out_len, out_str)
end

function LGBM_BoosterDumpModel(handle, start_iteration, num_iteration, buffer_len, out_len, out_str)
    ccall((:LGBM_BoosterDumpModel, liblightgbm), Cint, (BoosterHandle, Cint, Cint, Cint, Ptr{Cint}, Cstring), handle, start_iteration, num_iteration, buffer_len, out_len, out_str)
end

function LGBM_BoosterGetLeafValue(handle, tree_idx, leaf_idx, out_val)
    ccall((:LGBM_BoosterGetLeafValue, liblightgbm), Cint, (BoosterHandle, Cint, Cint, Ptr{Cdouble}), handle, tree_idx, leaf_idx, out_val)
end

function LGBM_BoosterSetLeafValue(handle, tree_idx, leaf_idx, val)
    ccall((:LGBM_BoosterSetLeafValue, liblightgbm), Cint, (BoosterHandle, Cint, Cint, Cdouble), handle, tree_idx, leaf_idx, val)
end

function LGBM_BoosterFeatureImportance(handle, num_iteration, importance_type, out_results)
    ccall((:LGBM_BoosterFeatureImportance, liblightgbm), Cint, (BoosterHandle, Cint, Cint, Ptr{Cdouble}), handle, num_iteration, importance_type, out_results)
end

function LGBM_NetworkInit(machines, local_listen_port, listen_time_out, num_machines)
    ccall((:LGBM_NetworkInit, liblightgbm), Cint, (Cstring, Cint, Cint, Cint), machines, local_listen_port, listen_time_out, num_machines)
end

function LGBM_NetworkFree()
    ccall((:LGBM_NetworkFree, liblightgbm), Cint, ())
end

function LGBM_NetworkInitWithFunctions(num_machines, rank, reduce_scatter_ext_fun, allgather_ext_fun)
    ccall((:LGBM_NetworkInitWithFunctions, liblightgbm), Cint, (Cint, Cint, Ptr{Cvoid}, Ptr{Cvoid}), num_machines, rank, reduce_scatter_ext_fun, allgather_ext_fun)
end

function LastErrorMsg()
    ccall((:LastErrorMsg, liblightgbm), Cstring, ())
end

function LGBM_SetLastError(msg)
    ccall((:LGBM_SetLastError, liblightgbm), Cvoid, (Cstring,), msg)
end
