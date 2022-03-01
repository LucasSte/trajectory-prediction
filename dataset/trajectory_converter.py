
class TrajectoryConverter:

    def __init__(self, look_back, look_forth, coord_std, coord_avg):
        self.look_back = look_back
        self.coord_std = coord_std
        self.coord_avg = coord_avg
        self.look_forth = look_forth

    def convert_single(self, x, y):
        last_pos = x[self.look_back-1, 0:2]
        last_pos = last_pos*self.coord_std[0:2] + self.coord_avg[0:2]
        y_local = y*self.coord_std[2:4] + self.coord_avg[2:4]
        y_local[0] = y_local[0] + last_pos
        for i in range(1, self.look_forth):
            y_local[i] = y_local[i-1] + y_local[i]

        return y_local

    def convert_batch(self, x, y):
        last_pos = x[:, self.look_back-1, 0:2]
        last_pos = last_pos*self.coord_std[0:2] + self.coord_avg[0:2]
        y_local = y*self.coord_std[2:4] + self.coord_std[2:4]
        y_local[:, 0] = y_local[:, 0] + last_pos

        for i in range(1, self.look_forth):
            y_local[:, i] = y_local[:, i-1] + y_local[:, i]

        return y_local
