import math
from datetime import datetime
import sys
import os
import random
import numpy as np
seed = 0
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
from scripts.utils.common_utils import CommonUtils,BaseScheduler


class TensorFlowScheduler(BaseScheduler):
    def __init__(self,name,drop_rate=None,change=None,switch_data=None):
        super().__init__(name,drop_rate=drop_rate,change=change,switch_data=switch_data)

    def loss_checker(self, loss_val,log_msg=None):
        if log_msg is None:
            if self.iter_count % 100 == 0:
                print(f"Iteration: {self.iter_count} Current Time cost:{datetime.now() - self.start_time} Loss:{loss_val}")
        if np.isnan(loss_val).any() or np.isinf(loss_val).any():
            if log_msg is not None:
                print(log_msg)
            CommonUtils.end_with_time(f"FINAL RESULTS: #{self.name}# Success to find NaN!  Iteration: <{self.iter_count}>",
                                      self.start_time)
            if not os.path.exists("/data/scripts"):
                os.makedirs("/data/scripts")
            with open("/data/scripts/results.txt", "a", encoding="utf-8") as file:
                file.write(f"#{self.name}# *Success* @{self.iter_count}@ <{datetime.now() - self.start_time}>\n")
                file.close()
            sys.exit(0)


class GradientSearcher(TensorFlowScheduler):

    def __init__(self, name,score_mthd=1, update_stragety='rate', drop_rate=None,change=None,switch_data=None):
        super().__init__(name,drop_rate=drop_rate,change=change,switch_data=switch_data)
        self.score_mthd = score_mthd
        self.get_score = CommonUtils.get_score1
        self.update_stragety = update_stragety
        # self.drop_rate = drop_rate
        self.batch_size = None
        self.bottom_k = None
        self.clip_cnt = None
        self.update_cnt = None
        self.iter_cnt = None
        self.useful_update_cnt = None
        self.pixels = None
        self.linear = False

    def build(self, batch_size, min_val=None, max_val=None):
        self.batch_size = batch_size
        self.bottom_k = int(batch_size * self.drop_rate)
        self.clip_cnt = [0 for _ in range(batch_size)]
        self.update_cnt = [0 for _ in range(batch_size)]
        self.iter_cnt = [0 for _ in range(batch_size)]
        self.useful_update_cnt = [0 for _ in range(batch_size)]
        print(f"Max value in x_train {max_val}")
        self.pixels = {'max': max_val, 'min': min_val}


    @staticmethod
    def iterative_grist(w, dw, pixle_boundary, update_stragety, update_list, clip_list, batch_size, useful_update_list, change, delta, delta_type, switch_data:bool):

        tmp_update = [0 for _ in range(batch_size)]
        tmp_clip = [0 for _ in range(batch_size)]
        kw = {'dw':dw,"change":change,"delta":delta,"type":delta_type}
        delta_val = CommonUtils.get_delta(kw)

        next_w = w - delta_val
        if switch_data:
            data_shape = w.shape[1:]
            pixel_num = CommonUtils.get_pixels_cnt(data_shape)
            next_w_flatten = next_w.flatten()
            modified_list = list(np.flatnonzero(abs(delta_val)))
            invalid_pixels = np.argwhere((next_w_flatten > pixle_boundary['max'])|(next_w_flatten<pixle_boundary['min'])).flatten()

            modified_img_idx = [CommonUtils.get_image_idx(pixel_local=pixel,image_shape=data_shape) for pixel in modified_list]
            invalid_img_idx = [CommonUtils.get_image_idx(pixel_local=pixel, image_shape=data_shape) for pixel in invalid_pixels]
            # update_list = [update_list[i] + 1 for i in modified_img_idx]
            for i in modified_img_idx:
                tmp_update[i] = tmp_update[i] + 1
            for i in invalid_img_idx:
                tmp_clip[i] = tmp_clip[i] + 1
            if update_stragety == 'useful':
                for img_idx in range(batch_size):
                    if tmp_update[img_idx] > tmp_clip[img_idx]:
                        useful_update_list[img_idx] += 1
            if update_stragety == 'rate':
                tmp_update1 = [n/pixel_num for n in tmp_update]
                tmp_clip1 = [n/pixel_num for n in tmp_clip]
                update_list = list(map(lambda x: x[0] + x[1], zip(tmp_update1, update_list)))
                clip_list = list(map(lambda x: x[0] + x[1], zip(tmp_clip1, clip_list)))
            return next_w,update_list,clip_list,useful_update_list
        else:
            return next_w, None, None, None

    def update_batch_data(self,session,monitor_var,feed_dict,input_data,):

        if self.decay:
            decay_change = max(self.change * pow(self.decay_rate,math.floor(self.iter_count / self.decay_steps)),self.change *1e-4)
        else:
            decay_change = self.change
        if 'delta' in monitor_var.keys():
            loss_val, obj_function_val, obj_grads_val,delta_val = session.run([monitor_var['loss'], monitor_var['obj_function'], monitor_var['obj_grad'],monitor_var['delta']],
                                                                 feed_dict=feed_dict)
        else:
            loss_val, obj_function_val, obj_grads_val = session.run(
                [monitor_var['loss'], monitor_var['obj_function'], monitor_var['obj_grad']],
                feed_dict=feed_dict)
            delta_val = None
        log_msg = f"INFO: Iter: {self.iter_count} Current Time cost:{datetime.now() - self.start_time} Loss: {loss_val} Object function value: {obj_function_val} Change step: {decay_change}"
        # if self.iter_count % 100 == 0:
        #     print(log_msg)
        #     print(f"obj grads: ({np.min(obj_grads_val)}, {np.max(obj_grads_val)})")
        self.loss_checker(loss_val=loss_val,log_msg=log_msg)
        batch_xs, update_cnt, clip_cnt, useful_update_cnt = GradientSearcher.iterative_grist(input_data, obj_grads_val,
                                                                                             pixle_boundary=self.pixels,
                                                                                             update_stragety=self.update_stragety,
                                                                                             update_list=self.update_cnt,
                                                                                             clip_list=self.clip_cnt,
                                                                                             batch_size=self.batch_size,
                                                                                             useful_update_list=self.useful_update_cnt,
                                                                                             change=decay_change, delta=delta_val, delta_type=self.delta_type, switch_data=self.switch_data)
        if self.switch_data:
            self.iter_cnt = [x + 1 for x in self.iter_cnt]
            scores = [self.get_score(i, clips=self.clip_cnt, updates=self.update_cnt, iters=self.iter_cnt, useful_updates=self.useful_update_cnt)
                      for i in range(self.batch_size)]
            scores_rank = list(np.argsort(np.array(scores)))[:self.bottom_k]
        else:
            scores_rank = None
        if self.iter_count % 100 == 0:
            print(log_msg)
            # print(f"obj grads: ({np.min(obj_grads_val)}, {np.max(obj_grads_val)}), input: ({np.min(batch_xs)}, {np.max(batch_xs)})")
        batch_xs = np.clip(batch_xs, self.pixels['min'], self.pixels['max'])
        return batch_xs,scores_rank

    def switch_new_data(self,new_data_dict:dict,old_data_dict:dict,scores_rank):
        # change batch data
        input_num = len(list(new_data_dict.keys()))
        new_batch_xs = new_data_dict['x']
        old_batch_xs = old_data_dict['x']
        if input_num == 2:
            new_batch_ys = new_data_dict['y']
            old_batch_ys = old_data_dict['y']
        if self.iter_count % 100 == 0:
            if self.switch_data:
                for i, image_idx in enumerate(scores_rank):
                    old_batch_xs[image_idx] = new_batch_xs[i].copy()
                    if input_num == 2:
                        old_batch_ys[image_idx] = new_batch_ys[i].copy()
                    self.iter_cnt[image_idx] = 0
                    self.clip_cnt[image_idx] = 0
                    self.update_cnt[image_idx] = 0
        if input_num == 2:
            return old_batch_xs, old_batch_ys
        else:
            return old_batch_xs, None

if __name__ == "__main__":
    pass