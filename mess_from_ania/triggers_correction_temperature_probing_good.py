#!/usr/bin/python
#!!!!!!!!!!!!!!!!!!!!!!!!!!!! RUN ONLY ONCE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

base_directory='Z:\\Data\\2023_08_14_bench\\long_runs\\mag_long_run_drift_000\\'
import os    
os.chdir(base_directory)
dir_name = os.path.basename(base_directory)

source_file = '_f.csv'
class Chunk:
    def __init__(self):
        self.number = None
        self.Bad_has_5 = False
        self.Trig_in2_start_1 = None
        #self.Trig_in2_finish_2= None
        self.rows = []

    def add_row(self, row):
        if self.number is not None and self.number != row[0]:
            return False
        self.number = row[0]
        self.rows.append(row)
        return True

    def compute(self, Trig_in2_index, Bad_index):
        for idx, row in enumerate(self.rows):
            if float(row[Trig_in2_index]) == 1 and self.Trig_in2_start_1 is None: # here trigger input value in V (soundpixx output)
                self.Trig_in2_start_1 = idx
            if float(row[Bad_index]) == 5: ##### put here the voltage of the arduino output trigger
                self.Bad_has_5 = True

        for idx, row in enumerate(self.rows):
           #row.append(1 if idx < self.Trig_in2_start_2 else 0)
            #row.append(0 if idx >= self.Trig_in2_start_2 and self.Bad_has_5 else 0)
            row.append(3 if idx >= self.Trig_in2_start_1 and float(row[Trig_in2_index]) == 1 and not self.Bad_has_5 else 0)

    def to_print(self):
        print(f"Chunk {self.number}: Trig_in2 {self.Trig_in2_start_1} Bad {self.Bad_has_5} len {len(self.rows)}")

    def to_array(self):
        return [row for row in self.rows]


class Data:
    def __init__(self):
        self.chunks = []

    def add(self, chunk):
        self.chunks.append(chunk)

    def compute(self, Trig_in2_idx, Bad_idx):
        for chunk in self.chunks:
            chunk.compute(Trig_in2_idx, Bad_idx)

    def to_print(self):
        for chunk in self.chunks:
            chunk.to_print()

    def to_array(self):
        ret = []
        for chunk in self.chunks:
            ret = ret + chunk.to_array()
        return ret


def read_file_csv(file):
    import csv
    f = open(file, newline='')
    reader = csv.reader(f, delimiter=',')

    col_names = next(reader)
    print(col_names)

    Trig_in2_idx, Bad_idx = (0, 0)

    for idx, name in enumerate(col_names):
        if name == 'Trig_in2':
            Trig_in2_idx = idx
        if name == 'Aux2_v': #######put here which channel are we using as the temperature sensing input
            Bad_idx = idx

    print(f"Trig_in2: {Trig_in2_idx}, Bad: {Bad_idx}")
    
    chunk = Chunk()
    data = Data()

    for row in reader:
        if not chunk.add_row(row):
            data.add(chunk)
            chunk = Chunk()
            chunk.add_row(row)
    data.add(chunk)

    data.to_print()

    data.compute(Trig_in2_idx, Bad_idx)
    data.to_print()

    with open(source_file, "w", newline='') as csv_file:
        result = [col_names + ['Stim']] + data.to_array()
        print('saving data, please wait')
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerows(result)
        f.close()




def main():

    read_file_csv(source_file)


if __name__ == "__main__":
    main()
