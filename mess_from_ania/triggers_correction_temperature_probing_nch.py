#!/usr/bin/python

#Hejka!
# dodac columny S0 S1 S2.
# S0 wstawia 1 dla kazdego chunka o danym numerze dopoki sie w Aux1 nie pojawi 5. potem podstawia 0 az do poczatku nastepnego chunka

# S1 podstawia 5 jesli w danym chunku sie w Aux 2 pojawia 5  i podstawia je of wiersza zielonego 5 (czyli kiedy w #aux 1 sie pierwsza 5 pojawia, az do konca chunka)

# S2 robi odwrotnie niz w S1 czyli jesli w Aux2 byla to w tym chunku wyrzuca wszedzie zera, a jesli nie bylo 5 to jak a Aux sie pojawia 5 to zaczyna podstawiac 5tki do konca chunka


base_directory='Z:\\Data\\2023_05_02_Zurich_dipole\\runs2\\1pTpp_1_000\\'
import os    
os.chdir(base_directory)
dir_name = os.path.basename(base_directory)

source_file = '_f.csv'
class Chunk:
    def __init__(self):
        self.number = None
        self.Aux1_v_has_5 = False
        self.Trig_in2_start_1 = None
        #self.Trig_in2_finish_2= None
        self.rows = []

    def add_row(self, row):
        if self.number is not None and self.number != row[0]:
            return False
        self.number = row[0]
        self.rows.append(row)
        return True

    def compute(self, Trig_in2_index, Aux1_v_index):
        for idx, row in enumerate(self.rows):
            if float(row[Trig_in2_index]) == 1 and self.Trig_in2_start_1 is None:
                self.Trig_in2_start_1 = idx
            if float(row[Aux1_v_index]) == 5:
                self.Aux1_v_has_5 = True

        for idx, row in enumerate(self.rows):
            #row.append(1 if idx < self.Trig_in2_start_2 else 0)
            #row.append(0 if idx >= self.Trig_in2_start_2 and self.Aux1_v_has_5 else 0)
            row.append(3 if idx >= self.Trig_in2_start_1 and float(row[Trig_in2_index]) == 1 and not self.Aux1_v_has_5 else 0)

    def to_print(self):
        print(f"Chunk {self.number}: Trig_in2 {self.Trig_in2_start_1} Aux1_v {self.Aux1_v_has_5} len {len(self.rows)}")

    def to_array(self):
        return [row for row in self.rows]


class Data:
    def __init__(self):
        self.chunks = []

    def add(self, chunk):
        self.chunks.append(chunk)

    def compute(self, Trig_in2_idx, Aux1_v_idx):
        for chunk in self.chunks:
            chunk.compute(Trig_in2_idx, Aux1_v_idx)

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

    Trig_in2_idx, Aux1_v_idx = (0, 0)

    for idx, name in enumerate(col_names):
        if name == 'Trig_in2':
            Trig_in2_idx = idx
        if name == 'Aux2_v':
            Aux1_v_idx = idx

    print(f"Trig_in2: {Trig_in2_idx}, Aux1_v: {Aux1_v_idx}")

    chunk = Chunk()
    data = Data()

    for row in reader:
        if not chunk.add_row(row):
            data.add(chunk)
            chunk = Chunk()
            chunk.add_row(row)
    data.add(chunk)

    data.to_print()

    data.compute(Trig_in2_idx, Aux1_v_idx)
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
