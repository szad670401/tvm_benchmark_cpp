#include <iostream>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <fstream>
#include <sys/time.h>

const int LOOP = 5000;


using namespace std;
static float getElapse(struct timeval *tv1,struct timeval *tv2)
{
    float t = 0.0f;
    if (tv1->tv_sec == tv2->tv_sec)
        t = (tv2->tv_usec - tv1->tv_usec)/1000.0f;
    else
        t = ((tv2->tv_sec - tv1->tv_sec) * 1000 * 1000 + tv2->tv_usec - tv1->tv_usec)/1000.0f;
    return t;
}

void benchmark(std::string so_lib_path,std::string graph_json,std::string graph_params,std::string model_name)
{
    tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(so_lib_path);
    std::cout<<"1"<<std::endl;
    std::ifstream json_in(graph_json, std::ios::in);
    std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
    json_in.close();
    std::ifstream params_in(graph_params, std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();
    TVMByteArray params_arr;
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();
    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = kDLGPU;
    int device_id = 4;
    tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib, device_type, device_id);
    DLTensor* x;
    int in_ndim = 4;
    int64_t in_shape[4] = {1, 3, 112, 112};

    TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);
    tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
    set_input("data", x);

    tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
    load_params(params_arr);

    tvm::runtime::PackedFunc run = mod.GetFunction("run");

    tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");

    struct timeval  tv1,tv2;

    float sum_time = 0;
    for(int i = 0 ; i<LOOP; i++)
    {
        gettimeofday(&tv1,NULL);
        run();
        tvm::runtime::NDArray res = get_output(0);
        gettimeofday(&tv2,NULL);
        float diff = getElapse(&tv1,&tv2);
        sum_time+=diff;

        printf("[%d] %s cost %fms\n",i,model_name.c_str(),diff);
    }
    printf("%s avg cost %f ms",model_name.c_str(),sum_time/LOOP);

    TVMArrayFree(x);
}

int main() {
    benchmark("r100_model/deploy_lib.so","r100_model/deploy_graph.json","r100_model/deploy_param.params","r100");



    return 0;
}
