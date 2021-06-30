import datetime
import configparser
import sys
import numpy as np
import math
import os


class CommonUtils:
    @staticmethod
    def get_random_learning_rate():
        # get random num from [0.0001,1] -> [10^-4,10^0]
        r = np.random.random()  # random = [0,1)
        r = r * (-4)
        r = pow(10, r)
        return round(r, 5)

    @staticmethod
    def get_score1(i, clips, updates, iters, useful_updates):
        return (updates[i] - clips[i] + 1) / (iters[i] + 1e-7)

    @staticmethod
    def get_score2(i, clips, updates, iters, useful_updates):
        return (updates[i] ** 2 + 1) / (iters[i] * clips[i] + 1e-7)

    @staticmethod
    def get_score3(i, clips, updates, iters, useful_updates):
        return (useful_updates[i]) / (iters[i] + 1e-7)

    @staticmethod
    def get_pixels_cnt(img_shape: tuple):
        pc = 1
        for shape in img_shape:
            pc *= shape
        return pc

    @staticmethod
    def get_image_idx(pixel_local: int, image_shape: tuple):
        pixels_cnt = CommonUtils.get_pixels_cnt(image_shape)
        return math.floor(pixel_local / pixels_cnt)

    @staticmethod
    def end_with_time(msg, start_time):
        print(f"{msg} Time cost: <{datetime.datetime.now() - start_time}>.")

    @staticmethod
    def get_delta(kw):
        delta_type = kw['type']
        if delta_type == 'grist':
            delta_val = kw['dw']
            delta_val[delta_val > 0] = 1
            delta_val[delta_val < 0] = -1
            delta_val *= kw['change']
            return delta_val
        # elif delta_type == 'newton':
        #     return kw['delta']
        else:
            raise Exception(f"No such delta type {delta_type}")

    @staticmethod
    def parse_msg(console_str):
        stats = console_str[console_str.find("<") + 1:console_str.find(">")]
        time_msg = console_str[console_str.rfind("<") + 1:console_str.rfind(">")]
        return stats, time_msg


class BaseScheduler:

    def __init__(self,name,drop_rate,change,switch_data):
        self.name = name
        self.start_time = datetime.datetime.now()
        dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        grist_cfg = configparser.ConfigParser()
        grist_cfg.read(os.path.join(dir_path,"config/experiments.config"))
        self.config_time = grist_cfg['parameter'].getfloat("time_limit") * 60
        self.time_limitation = datetime.timedelta(seconds=self.config_time)
        self.iter_count = 1
        self.change = grist_cfg['parameter'].getfloat("change") if change is None else change
        self.decay = grist_cfg['parameter'].getboolean("decay")
        self.decay_steps = grist_cfg['parameter'].getint("decay_steps")
        self.decay_rate = grist_cfg['parameter'].getfloat("decay_rate")
        self.drop_rate = grist_cfg['parameter'].getfloat("drop_rate") if drop_rate is None else drop_rate
        self.delta_type = grist_cfg['parameter'].get("delta_type")
        self.switch_data = grist_cfg['parameter'].getboolean("switch_data") if switch_data is None else switch_data
        self.print_parameters()

    def check_time(self):
        if datetime.datetime.now() - self.start_time > self.time_limitation:
            print(f"FINAL RESULTS: #{self.name}# Fail to find NaN! Iteration: <{self.iter_count}>. Time cost:<{self.time_limitation}>.")
            if not os.path.exists("/data/scripts"):
                os.makedirs("/data/scripts")
            with open("/data/scripts/results.txt", "a", encoding="utf-8") as file:
                file.write(f"#{self.name}# *Fail* @{self.iter_count}@ <{self.time_limitation}>\n")
                file.close()
            sys.exit(0)
        self.iter_count += 1

    def print_parameters(self):
        print("<-----Configuration----->")
        private_attrs = [a for a in self.__dir__() if not str(a).startswith("_")]
        for at in private_attrs:
            print(f"# {at}: {self.__getattribute__(at)}")
