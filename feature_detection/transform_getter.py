import datetime
import bisect

class TransformRecord:
    def __init__(self, date_str, x, y, z, rotation_x, rotation_y, rotation_z):
        self.date = datetime.datetime.strptime(date_str, "%Y/%m/%d %H:%M:%S:%f")
        self.position = (x, y, z)
        self.rotation = (rotation_x, rotation_y, rotation_z)
    
    def __str__(self):
        return f"Date: {self.date}, Position: {self.position}, Rotation: {self.rotation}"

    @staticmethod
    def from_string(line):
        parts = line.strip().split(",")
        date_str = parts[0]
        x, y, z = map(float, parts[1:4])
        rotation_x, rotation_y, rotation_z = map(float, parts[4:7])
        return TransformRecord(date_str, x, y, z, rotation_x, rotation_y, rotation_z)

class TransformGetter:
    def __init__(self, filename):
        self.transforms = self.load_transforms(filename)
    
    @staticmethod
    def load_transforms(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
        transforms = [TransformRecord.from_string(line) for line in lines]
        transforms.sort(key=lambda tr: tr.date)
        return transforms

    def binary_search(self, target_date):
        start = 0
        end = len(self.transforms) - 1
        while start <= end:
            mid = (start + end) // 2
            if self.transforms[mid].date < target_date:
                start = mid + 1
            elif self.transforms[mid].date > target_date:
                end = mid - 1
            else:
                return mid
        return start

    def get_transform(self, date_str):
        target_date = datetime.datetime.strptime(date_str, "%d.%m.%Y_%H-%M-%S")
        index = self.binary_search(target_date)
        if index == 0:
            return self.transforms[0]
        if index == len(self.transforms):
            return self.transforms[-1]
        before = self.transforms[index - 1]
        after = self.transforms[index]
        if after.date - target_date < target_date - before.date:
           return after
        else:
           return before


# example on how to use the class from outside
if __name__ == "__main__":
    # change the path to the file you want to load
    transform_getter = TransformGetter('Simulation_input/2023.6.9_15.14/TransformRecord.txt')
    # simply query the transform for a given date (which is the same as the date in image name)
    print(transform_getter.get_transform("09.06.2023_14-51-11"))
