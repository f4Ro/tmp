from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(10)

import tensorflow as tf
from models.crbm_tf_1.CRBM import CRBM
import numpy as np
import logging
import sys
import matplotlib.pyplot as plt
from models.crbm_tf_1.mnist_example import mnist

plot_enabled = True

if __name__ == '__main__':

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # img = np.divide(np.array(mnist), 255).astype('float32')
    from data_preprocessing.berkley_lab_data import read_and_preprocess_data
    x_train, x_test = read_and_preprocess_data(
        should_smooth=False,
        smoothing_window=100,
        sequence_length=120,
        cut_off_min=5,
        cut_off_max=45,
        should_scale=True,
        data_path="datasets/data.txt",
        batch_size=1,
        motes_train=[7],
        motes_test=[7]
    )

    img = x_train[:1, :, :].reshape(120,1).astype('float32')
    # mean = np.mean(inp)

    # test it..
    filter_shape = (1, 1)
    visible_shape = (120, 1)
    k = 500
    params_id = 'test1'

    # testing runnability of crbm
    crbm = CRBM(filter_shape, visible_shape, k, params_id)
    hidden = crbm.generate_hidden_units(np.expand_dims(img,0))
    rec = crbm.generate_visible_units(hidden, 1)
    print('CRBM running ok')

    #### test Trainer
    crbm = CRBM(filter_shape, visible_shape, k, params_id)
    trainer = crbm.Trainer(crbm)
    img_fitted = np.expand_dims(img,0)

    from shared_code.per_rms_diff import per_rms_diff
    hidden = crbm.generate_hidden_units(img_fitted)
    rec = crbm.generate_visible_units(hidden, 1)
    prms_before = per_rms_diff(img, rec)

    for i in range(300):
        if i % 20 == 0:
            print('Epoch: ', i)
            trainer.train(img_fitted,sigma=0.01, lr=0.0001, verbose=True)

    hidden = crbm.generate_hidden_units(img_fitted)
    rec = crbm.generate_visible_units(hidden, 1)
    prms_after = per_rms_diff(img, rec)
    print(f'PRMS {prms_before} -> {prms_after}')

    if plot_enabled:
        hidden = crbm.generate_hidden_units(img_fitted)
        rec = crbm.generate_visible_units(hidden, 1)
        plt.switch_backend('WebAgg')
        plt.figure()
        plt.suptitle('original image')
        plt.plot(img)
        plt.figure()
        plt.suptitle('reconstructed image')
        plt.plot(rec[0,:,:])
        plt.show()

    #### test Saver
    # Saver = CRBM.Saver
    # save_path = 'test/save_variables/params_1/w'
    # Saver.save(crbm, save_path, 0)

    # crbm = CRBM(filter_shape, visible_shape, k, params_id)
    # Saver.restore(crbm, save_path+'-0')
    # # resume training with
    # for _ in range(40):
    #     trainer.train(img_fitted,sigma=0.01, lr=0.000001)

    # #### test Trainer with summary.
    # summary_dir = 'test/summaries/params_1/'
    # trainer = crbm.Trainer(crbm, summary_enabled=True, summary_dir=summary_dir)
    # for _ in range(10):
    #     trainer.train(img_fitted,sigma=0.01, lr=0.000001)
    #     print('see {0} with tensorboard'.format(summary_dir))

    # print('Ctrl C to exit')
    # while True:
    #     pass
