import paddle
import time


if __name__ == "__main__":
    batch_size = 16
    num_joint = 21

    index_x = paddle.randint(low=0, high=64, shape=(batch_size, num_joint * 2))
    idx_zy = paddle.randint(low=0, high=64, shape=(batch_size, num_joint * 2, 64))
    shape = (-1, 1) if batch_size > 1 else (-1, )

    for j in range(100):
        start = time.time()
        idx = paddle.concat((
            paddle.arange(0, batch_size).reshape(shape).expand_as(index_x).reshape((-1, 1)),
            paddle.arange(0, num_joint * 2).expand_as(index_x).reshape((-1, 1)),
            index_x.reshape((-1, 1))
        ),
            axis=1)
        joint_y_con = paddle.gather_nd(idx_zy, idx).reshape((batch_size, num_joint * 2, 1))
        print(f"concat&gather_nd time cost: {time.time() - start:.2f}s")

        start = time.time()
        joint_y_for = paddle.zeros(shape=[idx_zy.shape[0], idx_zy.shape[1], 1], dtype=idx_zy.dtype)
        for i in range(batch_size):
            for idx, ix in enumerate(index_x[i]):
                joint_y_for[i, idx, 0] = idx_zy[i, idx, ix]
        print(f"for loop time cost: {time.time() - start:.2f}s")
        print(f"diff: {paddle.sum(joint_y_con - joint_y_for).cpu().numpy()}")