#include "model_helper.h"

//#include <experimental/filesystem>
#include <fstream>
#include <string>
#include <sys/time.h>
#include <iostream>
#include <ros/ros.h>

static std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();

/***********************************************************************/
MobilenetV1::MobilenetV1()
{
}

MobilenetV1::~MobilenetV1()
{
    // must release before edgetpu_context release
    interpreter_.reset();
}

void MobilenetV1::init(const string &model_path)
{
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    interpreter_ =BuildEdgeTpuInterpreter(*model_, edgetpu_context.get());
}

int MobilenetV1::inference(const std::vector<uint8_t> &input, const std::unique_ptr<tflite::Interpreter> &interpreter)
{
    uint8_t *input_tf = interpreter->typed_input_tensor<uint8_t>(0);
    std::memcpy(input_tf, input.data(), input.size());

    interpreter->Invoke();

    std::vector<float> output_data;
    const auto &output_indices = interpreter->outputs();
    //cout<<output_indices<<endl;
    const int num_outputs = output_indices.size();
    int out_idx = 0;
    for (int i = 0; i < num_outputs; ++i)
    {
        const auto *out_tensor = interpreter->tensor(output_indices[i]);
        assert(out_tensor != nullptr);
        if (out_tensor->type == kTfLiteUInt8)
        {
            const int num_values = out_tensor->bytes;
            //cout<<"num_values:"<<num_values<<endl;
            output_data.resize(out_idx + num_values);
            const uint8_t *output = interpreter->typed_output_tensor<uint8_t>(i);
            for (int j = 0; j < num_values; ++j)
            {
                //cout<<"output[j]:"<<(uint8_t)(output[j])<<endl;
                //cout<<"zero_point:"<<out_tensor->params.zero_point<<endl;
                //cout<<"scale:"<<out_tensor->params.scale<<endl;
                output_data[out_idx++] = (output[j] - out_tensor->params.zero_point) * out_tensor->params.scale;
            }
        }
        else if (out_tensor->type == kTfLiteFloat32)
        {
            const int num_values = out_tensor->bytes / sizeof(float);
            output_data.resize(out_idx + num_values);
            const float *output = interpreter->typed_output_tensor<float>(i);
            for (int j = 0; j < num_values; ++j)
            {
                output_data[out_idx++] = output[j];
            }
        }
        else
        {
            std::cerr << "Tensor " << out_tensor->name
                      << " has unsupported output type: " << out_tensor->type
                      << std::endl;
        }
    }

    //cout<<"output_data size:"<<output_data.size()<<endl;
    for (auto data : output_data)
    {
        //cout<<"prob:"<<data<<endl;
    }

    auto it = std::max_element(output_data.begin(), output_data.end());
    int cls_idx = std::distance(output_data.begin(), it);
    //std::cout << "index: "<< cls_idx << " value: " << *it<< std::endl;

    return cls_idx;
}

int MobilenetV1::inference(const std::vector<uint8_t> &input)
{
    return inference(input, interpreter_);
}

/***********************************************************************************/
MobilenetV1SSD::MobilenetV1SSD()
{
}

MobilenetV1SSD::~MobilenetV1SSD()
{
    interpreter_.reset();
}

void MobilenetV1SSD::init(const string &model_path)
{
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    interpreter_ = BuildEdgeTpuInterpreter(*model_, edgetpu_context.get());
}

vector<BBoxInfo> MobilenetV1SSD::inference(const std::vector<uint8_t> &input)
{
    uint8_t *input_tf = interpreter_->typed_input_tensor<uint8_t>(0);
    std::memcpy(input_tf, input.data(), input.size());

    interpreter_->Invoke();

    //std::vector<float> output_data;
    vector<vector<float>> result;
    const auto &output_indices = interpreter_->outputs();
    const int num_outputs = output_indices.size();

    int out_idx = 0;
    for (int i = 0; i < num_outputs; ++i)
    {
        vector<float> tensor_i;
        const auto *out_tensor = interpreter_->tensor(output_indices[i]);
        assert(out_tensor != nullptr);
        if (out_tensor->type == kTfLiteUInt8)
        {
            const int num_values = out_tensor->bytes;
            //cout<<"the "<<i<<" tensor size:"<<num_values<<endl;

            tensor_i.resize(num_values);
            const uint8_t *output = interpreter_->typed_output_tensor<uint8_t>(i);
            for (int j = 0; j < num_values; ++j)
            {
                //cout<<"output[j]:"<<(uint8_t)(output[j])<<endl;
                //cout<<"zero_point:"<<out_tensor->params.zero_point<<endl;
                //cout<<"scale:"<<out_tensor->params.scale<<endl;
                tensor_i[j] = (output[j] - out_tensor->params.zero_point) * out_tensor->params.scale;
            }

            result.emplace_back(tensor_i);
        }
        else if (out_tensor->type == kTfLiteFloat32)
        {
            const int num_values = out_tensor->bytes / sizeof(float);
            //cout<<"the "<<i<<" tensor size:"<<num_values<<endl;

            tensor_i.resize(num_values);
            const float *output = interpreter_->typed_output_tensor<float>(i);
            for (int j = 0; j < num_values; ++j)
            {
                tensor_i[j] = output[j];
            }

            result.emplace_back(tensor_i);
        }
        else
        {
            std::cerr << "Tensor " << out_tensor->name
                      << " has unsupported output type: " << out_tensor->type
                      << std::endl;
        }
    }

    vector<BBoxInfo> ret;
    int n = lround(result[3][0]); //box number
    for (int i = 0; i < n; ++i)
    {
        int id = lround(result[1][i]);
        float score = result[2][i];
        if (score < score_threshold_)
            continue;

        float ymax = std::max(static_cast<float>(0.0), result[0][4 * i]);
        float xmax = std::max(static_cast<float>(0.0), result[0][4 * i + 1]);
        float ymin = std::min(static_cast<float>(1.0), result[0][4 * i + 2]);
        float xmin = std::min(static_cast<float>(1.0), result[0][4 * i + 3]);

        ret.emplace_back(BBoxInfo({xmin, ymin, xmax, ymax, id, score}));
    }

    for (auto b : ret)
    {
        b.output();
    }

    return ret;
}

std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(
    const tflite::FlatBufferModel &model,
    edgetpu::EdgeTpuContext *edgetpu_context)
{
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
    std::unique_ptr<tflite::Interpreter> interpreter;
    if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk)
    {
        std::cerr << "Failed to build interpreter." << std::endl;
    }
    // Bind given context with interpreter.
    interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
    interpreter->SetNumThreads(1);
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "Failed to allocate tensors." << std::endl;
    }

    return interpreter;
};
