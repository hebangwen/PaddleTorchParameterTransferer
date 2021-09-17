import os

import paddle
from paddle.inference import Config
from paddle.inference import create_predictor



class Predictor(object):
    def __init__(self,
                model_dir,
                use_gpu=False,
                run_mode='fluid',
                use_dynamic_shape=False,
                trt_min_shape=1,
                trt_max_shape=1280,
                trt_opt_shape=640,
                min_subgraph_size=3,
                threshold=0.5):
        self.predictor = self.load_predictor(
            model_dir,
            run_mode=run_mode,
            min_subgraph_size=min_subgraph_size,
            use_gpu=use_gpu,
            use_dynamic_shape=use_dynamic_shape,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape)
        self.threshold = threshold

    def load_predictor(self,
                       model_dir,
                       run_mode='fluid',
                       batch_size=1,
                       use_gpu=False,
                       min_subgraph_size=3,
                       use_dynamic_shape=False,
                       trt_min_shape=1,
                       trt_max_shape=1280,
                       trt_opt_shape=640):
        if not use_gpu and not run_mode == 'fluid':
            raise ValueError(
                "Predict by TensorRT mode: {}, expect use_gpu==True, but use_gpu == {}"
                    .format(run_mode, use_gpu))
        use_calib_mode = True if run_mode == 'trt_int8' else False
        config = Config(
            os.path.join(model_dir, 'model.pdmodel'),
            os.path.join(model_dir, 'model.pdiparams'))
        precision_map = {
            'trt_int8': Config.Precision.Int8,
            'trt_fp32': Config.Precision.Float32,
            'trt_fp16': Config.Precision.Half
        }
        if use_gpu:
            # initial GPU memory(M), device ID
            config.enable_use_gpu(200, 0)
            # optimize graph and fuse op
            config.switch_ir_optim(True)
        else:
            config.disable_gpu()

        if run_mode in precision_map.keys():
            config.enable_tensorrt_engine(
                workspace_size=1 << 10,
                max_batch_size=batch_size,
                min_subgraph_size=min_subgraph_size,
                precision_mode=precision_map[run_mode],
                use_static=False,
                use_calib_mode=use_calib_mode)

            if use_dynamic_shape:
                print('use_dynamic_shape')
                min_input_shape = {'image': [1, 3, trt_min_shape, trt_min_shape]}
                max_input_shape = {'image': [1, 3, trt_max_shape, trt_max_shape]}
                opt_input_shape = {'image': [1, 3, trt_opt_shape, trt_opt_shape]}
                config.set_trt_dynamic_shape_info(min_input_shape, max_input_shape,
                                                  opt_input_shape)
                print('trt set dynamic shape done!')

        # disable print log when predict
        config.disable_glog_info()
        # enable shared memory
        config.enable_memory_optim()
        # disable feed, fetch OP, needed by zero_copy_run
        config.switch_use_feed_fetch_ops(False)
        predictor = create_predictor(config)
        return predictor
    
    def predict(self, inputs, warmup=0, repeats=1):
        input_names = self.predictor.get_input_names()
        print(input_names)
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs)
        
        for i in range(warmup):
            self.predictor.run()
            output_names = self.predictor.get_output_names()

        import time
        for i in range(repeats):
            t = time.time()
            self.predictor.run()
            print(f"inference time: {(time.time() - t) * 1000:.2f}ms")
            output_names = self.predictor.get_output_names()
    

if __name__ == "__main__":
    model_dir = "output/interhand_trt"
    inputs = paddle.randn((1, 3, 256, 256))
    predictor = Predictor(
        model_dir,
        run_mode="trt_fp32",
        threshold=0.45,
        use_gpu=True,
        min_subgraph_size=15,
    )

    predictor.predict(inputs, repeats=100)