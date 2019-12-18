BasicEngine::BasicEngine(const std::string& model_path) {
  BasicEngineNativeBuilder builder(model_path);
  LOG_IF(FATAL, builder(&engine_) == kEdgeTpuApiError)
      << builder.get_error_message();
}

BasicEngineNativeBuilder::BasicEngineNativeBuilder(
    const std::string& model_path)
    : model_path_(model_path), device_path_("") {
  read_from_file_ = true;
  error_reporter_ = absl::make_unique<EdgeTpuErrorReporter>();
}

EdgeTpuApiStatus BasicEngineNativeBuilder::operator()(
    std::unique_ptr<BasicEngineNative>* engine) {
  EDGETPU_API_REPORT_ERROR(
      error_reporter_, !engine,
      "Null output pointer passed to BasicEngineNativeBuilder!");
  (*engine) = absl::make_unique<BasicEngineNative>();
  if (read_from_file_) {
    EDGETPU_API_REPORT_ERROR(
        error_reporter_,
        (*engine)->Init(model_path_, device_path_) != kEdgeTpuApiOk,
        (*engine)->get_error_message());
  } else {
    EDGETPU_API_REPORT_ERROR(error_reporter_, !model_, "model_ is nullptr!");
    EDGETPU_API_REPORT_ERROR(
        error_reporter_,
        (*engine)->Init(std::move(model_), resolver_.get()) != kEdgeTpuApiOk,
        (*engine)->get_error_message());
  }
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus BasicEngineNative::Init(const std::string& model_path,
                                         const std::string& device_path) {
  EDGETPU_API_ENSURE_STATUS(BuildModelFromFile(model_path));
  EDGETPU_API_ENSURE_STATUS(InitializeEdgeTpuResource(device_path));
  EDGETPU_API_ENSURE_STATUS(CreateInterpreterWithResolver(nullptr));
  EDGETPU_API_ENSURE_STATUS(InitializeInputAndOutput());
  is_initialized_ = true;
  return kEdgeTpuApiOk;
}


  model_ = tflite::FlatBufferModel::BuildFromFile(model_path_.c_str(),
                                                  error_reporter_.get());

 EdgeTpuResourceManager::GetSingleton()->GetEdgeTpuResource(
                &edgetpu_resource_)


EdgeTpuApiStatus BasicEngineNative::CreateInterpreterWithResolver(
    BuiltinOpResolver* resolver) {
  BuiltinOpResolver new_resolver;

  BuiltinOpResolver* effective_resolver =
      (resolver == nullptr ? &new_resolver : resolver);
  effective_resolver->AddCustom(edgetpu::kCustomOp,
                                edgetpu::RegisterCustomOp());
  effective_resolver->AddCustom(kPosenetDecoderOp, RegisterPosenetDecoderOp());

  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder interpreter_builder(
      model_->GetModel(), *effective_resolver, error_reporter_.get());
  if (interpreter_builder(&interpreter_) != kTfLiteOk) {
    error_reporter_->Report("Error in interpreter initialization.");
    return kEdgeTpuApiError;
  }
  // Bind given context with interpreter.
  interpreter_->SetExternalContext(kTfLiteEdgeTpuContext,
                                   edgetpu_resource_->context());
  interpreter_->SetNumThreads(1);
  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    error_reporter_->Report("Failed to allocate tensors.");
    return kEdgeTpuApiError;
  }

  EDGETPU_API_ENSURE(interpreter_);
  return kEdgeTpuApiOk;
}