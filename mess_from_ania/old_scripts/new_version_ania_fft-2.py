#!/usr/bin/python
fn = '_g_r_os'  # file to read without .csv extension
base_directory='Z:\\Data\\2023_06_09_brain\\Z\\sig\\brain6_25BW_700\\'
import os    
os.chdir(base_directory)
dir_name = os.path.basename(base_directory)

class Options:
    def __init__(self):
        self.headers = True
        self.input_delimiter = ','
        self.input_directory = '.\\'
        self.output_name = str(fn)
        self.averages = {
            'head_len': int(0.2*837.1),  # baseline correction - number of points used to calculate baseline
            'tail_len': int(0.2*837.1),
            'drop': False,
            'tail_head_diff': 12e-12
            }
        self.averages_for_x_ms = {  # this is a new feature i added in order to calculate average inside of each chunk for the first X rows
            'aver_len': 650
            }
        self.filter = {
            'chunk_min': 0,
            'chunk_max': 50,
            'drop': False
            }
        self.choose_freq = {  #(ktory wezmie pod uwage 'trigger_V' kolumne i odrzuci chunki co leza poza zakresem)
            'trigger_min': -4,  # trigger value usually should be +5V for freq1 and -5V for freq2
            'trigger_max': -2,
            'drop': False,
            }
        self.error_deg = {
            'head_len': 130,  # baseline correction - number of points used to calculate baseline
            'tail_len': 200,
            'drop': False,
            'tail_head_diff': 0.02
        }
        self.functions = [
            {
                'label': 'Standard deviation',
                'callback': fun_std,
                'filter': 800,
                'drop': False,
                'drop_corrected': False
            },
            {
                'label': 'Arithmetic mean',
                'callback': fun_mean,
                'filter_min': -0.2,
                'filter_max': 0.2,
                'drop': False,  # applies filter to values which are not "corrected" shifted by the initial offset determined by head_len
                'drop_corrected': False,  # filter is applied to the corrected values
            }]


def fun_std(in_array):
    import numpy
    res = numpy.std(in_array)
    return res


def fun_mean(in_array):
    import numpy
    res = numpy.mean(in_array)
    return res


# dwie nowe funkcje
def fun_fft(in_array):
    from scipy.fftpack import fft
    res = fft(in_array)
    return res


def fun_psd(in_array):
    import numpy
    import scipy
    import scipy.fftpack
    pass


# x=data1['time']

# y=np.array(data1['value_pT'])

# y_fft=sp.fftpack.fft(y2)

# y_psd=np.abs(y_fft)**2
# freq=3.823/3200
# fftfreq=sp.fftpack.fftfreq(len(y_psd),sfreq)
# i=fftfreq>0
    
class Chunk:
    def __init__(self):
        self.number = None
        self.base_timestamp = None
        self.timestamps = []
        self.values = []
        self.time_s = []
        self.values_pt = []
        self.triggers = []
        self.error_degs = []
        
        self.corrected_values = []

        self.dropped = False
        self.head_average = None
        self.tail_average = None
        self.average_for_x_ms = None
        self.difference = None
        self.error_deg = None

        self.functions = []
        self.functions_corrected = []
        self.mean_for_x_ms = []
        self.fft_values = []
        self.fft_corrected_values = []

    def add_row(self, number: int, value: float, time: float, B_T (pT): float,  error_deg: float, Aux1_v: float, Aux2_v: float,Trig_in2: float) \
            -> bool:
        if self.number is not None and self.number != number:
            self.compute_averages()
            return False
        self.number = number
        self.values.append(value*71)
        self.time_s.append(time_s)
        self.values_pt.append(value_pt)
        self.triggers.append(trigger)
        if options.choose_freq['drop'] and \
                (
                        trigger > options.choose_freq['trigger_min']
                        and trigger < options.choose_freq['trigger_max']
                ):
            self.dropped = True
        self.error_degs.append(error_deg)

        return True
        
    def compute_averages(self):
        import math
        self.head_average = sum(self.values[:options.averages['head_len']]) / options.averages['head_len']
        self.tail_average = sum(self.values[-options.averages['tail_len']:]) / options.averages['tail_len']
        self.difference = self.head_average - self.tail_average

        if options.averages['drop'] \
                and math.fabs(self.head_average - self.tail_average) > options.averages['tail_head_diff']:
            # print('Chunk %s was dropped because of diff in average of head and tail' % self.number)
            self.dropped = True

    def compute_error_deg(self):
        import math
        self.head_average = sum(self.error_degs[:options.averages['head_len']]) / options.error_deg['head_len']
        self.tail_average = sum(self.error_degs[-options.averages['tail_len']:]) / options.error_deg['tail_len']
        self.difference = self.head_average - self.tail_average

        if options.error_deg['drop'] \
                and math.fabs(self.head_average - self.tail_average) > options.error_deg['tail_head_diff']:

            # print('Chunk %s was dropped because of diff in average of head and tail' % self.number)
            self.dropped = True
   
    def compute_averages_for_x_ms(self):
        self.average_for_x_ms = sum(self.values[:options.averages_for_x_ms['aver_len']]) / options.averages_for_x_ms['aver_len']
        # print('average of chunk %s over '  % self.number, options.averages_for_x_ms['aver_len'], 'ms', self.average_for_x_ms)
        # print(self.mean_for_x_ms)
        
    def compute(self):
        import math
        # Baseline correction: Computing corrected values - reduce values by head average
        
        if self.head_average is None:
            self.compute_averages()

        if self.error_deg is None:
            self.compute_error_deg()
            
        if self.average_for_x_ms is None:     
            self.compute_averages_for_x_ms()
        
        self.corrected_values = [round(val - self.head_average, 5) for val in self.values]

        for fun in options.functions:
            fun_val = fun['callback'](self.values)
            if fun['drop']:
                if 'filter' in fun and math.fabs(fun_val) > fun['filter'] and not self.dropped:
                    # print('Chunk no %s marked dropped because of filter %s.' % (self.number, fun['label'], ))
                    self.dropped = True
                if 'filter_min' in fun and fun_val < fun['filter_min'] and not self.dropped:
                    # print('Chunk no %s marked dropped because of filter %s.' % (self.number, fun['label'], ))
                    self.dropped = True
                if 'filter_max' in fun and fun_val > fun['filter_max'] and not self.dropped:
                    # print('Chunk no %s marked dropped because of filter %s.' % (self.number, fun['label'], ))
                    self.dropped = True
            self.functions.append(fun_val)

            fun_val = fun['callback'](self.corrected_values)
            if fun['drop_corrected']:
                if 'filter' in fun and math.fabs(fun_val) > fun['filter'] and not self.dropped:
                    # print('Chunk no %s marked dropped because of filter %s.' % (self.number, fun['label'], ))
                    self.dropped = True
                if 'filter_min' in fun and fun_val < fun['filter_min'] and not self.dropped:
                    # print('Chunk no %s marked dropped because of filter %s.' % (self.number, fun['label'], ))
                    self.dropped = True
                if 'filter_max' in fun and fun_val > fun['filter_max'] and not self.dropped:
                    # print('Chunk no %s marked dropped because of filter %s.' % (self.number, fun['label'], ))
                    self.dropped = True
            self.functions_corrected.append((fun['callback'](self.corrected_values)))

        self.fft_values = fun_fft(self.values)
        self.fft_corrected_values = fun_fft(self.corrected_values)

    def to_array(self):
        return self.values

    def to_array_corrected(self):
        return self.corrected_values

    def output(self):
        return [[self.number, self.values[i], self.time[i], 
                 self.values_pt[i], self.triggers[i], self.error_degs[i]] for i in range(len(self.values))]


class Data:
    def __init__(self):
        self.chunks = []
        self.chunks_length = None
        self.timestamps_averages = []
        self.timestamps_averages_corrected = []

    def add_chunk(self, chunk):
        chunk_len = len(chunk.values)
        if self.chunks_length is not None \
                and self.chunks_length != chunk_len \
                and not chunk.dropped:
            print('Chunk no %s marked dropped because of difference in chunk length. Is %s, should be %s.'
                  % (chunk.number, chunk_len, self.chunks_length))
            chunk.dropped = True

        if False and self.chunks_length is not None \
                and self.chunks[0].timestamps != chunk.timestamps \
                and not chunk.dropped:
            print('Chunk no %s marked dropped because of difference in timestamps.' % chunk.number)
            chunk.dropped = True

        if not chunk.dropped \
                and (chunk.number < options.filter['chunk_min'] or chunk.number > options.filter['chunk_max']) \
                and options.filter['drop']:
            print('Chunk no %s marked dropped because of difference filter.' % chunk.number)
            chunk.dropped = True

        if not chunk.dropped \
                and (chunk.number < options.filter['chunk_min'] or chunk.number > options.filter['chunk_max']) \
                and options.filter['drop']:
            print('Chunk no %s marked dropped because of difference filter.' % chunk.number)
            chunk.dropped = True
            
        if self.chunks_length is None:
            self.chunks_length = chunk_len

        # print('Saving chunk no: %s' % chunk.number)
        self.chunks.append(chunk)
        
    def compute_chunks(self):
        import numpy

        for chunk in self.chunks:
            chunk.compute()

        temp_chunks = [chunk.to_array() for chunk in self.chunks if not chunk.dropped]
        self.timestamps_averages = [round(float(sum(col)) / len(col), 5) for col in zip(*temp_chunks)]
        temp_chunks = [chunk.to_array_corrected() for chunk in self.chunks if not chunk.dropped]
        self.timestamps_averages_corrected = [round(float(sum(col)) / len(col), 5) for col in zip(*temp_chunks)]

        num = 0
        for chunk in self.chunks:
            if not chunk.dropped:
               num = num + 1

        # print(f"Done computing chunks. {num} chunks remaining")

    def save_to_file(self, filename):
        import csv
        import numpy
        import scipy

        print("Saving private Chunk")

        with open(filename + '_org.csv', "w", newline='') as csv_file:
            result_2 = [chunk.output() for chunk in self.chunks if not chunk.dropped]

            result_1 = [["chunk", "timestamp", "value", "time", "value_pT", "trigger_V", "error_deg"]] \
                       + [item for sublist in result_2 for item in sublist]
            writer = csv.writer(csv_file, delimiter = ',')
            writer.writerows(result_1)


        with open(filename + '_all.csv', "w", newline='') as csv_file:
            result_1 = [['timestamps\\chunk'] + self.chunks[0].timestamps + [fun['label'] for fun in options.functions]] \
                        + [[row.number] + row.values + row.functions for row in self.chunks if not row.dropped] \
                        + [['average'] + self.timestamps_averages + ['' for fun in options.functions]]
            result = numpy.transpose(result_1)
            print('chunks saved:', numpy.size(result,1)-2)
            print('whole matrix',numpy.size(result))
            noise_pkpk=numpy.amax(self.timestamps_averages,axis=0)-numpy.amin(self.timestamps_averages,axis=0)
            print('noise_pk_pk:',noise_pkpk)
            print('noise rms (1 sdev):', numpy.std(self.timestamps_averages, axis=0))
            print('noise mean:', numpy.mean(self.timestamps_averages, axis=0))
            from scipy import stats
            print('noise stdev of the mean (SEM):', scipy.stats.sem(self.timestamps_averages, axis=0, ddof=0))
            writer = csv.writer(csv_file, delimiter = ',')
            writer.writerows(result)

        """
        with open(filename + '_fft.csv', "w", newline='') as csv_file:
            result_1 = [['timestamp\\chunk'] + self.chunks[0].timestamps] + [[row.number] + row.fft_values for row in self.chunks if not row.dropped]
            result = numpy.transpose(result_1)

            writer = csv.writer(csv_file, delimiter=',')
            writer.writerows(result)

        with open(filename + '_fft_corrected.csv', "w", newline='') as csv_file:
            result_1 = [['timestamp\\chunk'] + self.chunks[0].timestamps] \
                       + [[row.number] + row.fft_corrected_values for row in self.chunks if not row.dropped]
            result = numpy.transpose(result_1)
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerows(result)
        """
        # with open(filename + '_meg.csv', "w", newline='') as csv_file:
        #     result_1 = [['time_s\\chunk'] + self.chunks[0].time_s + [fun['label'] for fun in options.functions]] \
        #                 + [[row.number] + row.values + row.functions for row in self.chunks if not row.dropped] \
        #                 + [['average'] + self.timestamps_averages + ['' for fun in options.functions]]

        #     result = numpy.transpose(result_1)
        #     print('chunks saved:', numpy.size(result,1)-2)
        #     print('whole matrix',numpy.size(result))
        #     noise_pkpk=numpy.amax(self.timestamps_averages,axis=0)-numpy.amin(self.timestamps_averages,axis=0)
        #     print('noise_pk_pk:',noise_pkpk)
        #     print('noise rms (1 sdev):', numpy.std(self.timestamps_averages, axis=0))
        #     print('noise mean:', numpy.mean(self.timestamps_averages, axis=0))
        #     from scipy import stats
        #     print('noise stdev of the mean (SEM):', scipy.stats.sem(self.timestamps_averages, axis=0, ddof=0))
        #     writer = csv.writer(csv_file, delimiter = ',')
        #     writer.writerows(result)

        with open(filename + '_corrected.csv', "w", newline='') as csv_file:
            result_1 = [['timestamps\\chunk'] + self.chunks[0].timestamps + [fun['label'] for fun in options.functions]] \
                        + [[row.number] + row.corrected_values + row.functions_corrected for row in self.chunks if not row.dropped] \
                        + [['average'] + self.timestamps_averages_corrected + ['' for fun in options.functions]]
            result = numpy.transpose(result_1)

            writer = csv.writer(csv_file, delimiter = ',')
            writer.writerows(result)
            ##############################
            # tu chce zapisac liste samych wynikow, nr chunka i ta nowa srednia, albo w kol 1 nowa srednia, w kol 2 srednia average header. w 3 std. w 4 srednia wszystkiego
            # ale juz bez wartosci
        with open(filename + '_stats_only.csv', "w", newline='') as csv_file:
       
            result_1 = [['chunk'] + ['average for ' +  str(options.averages_for_x_ms['aver_len']) +' pts'] + ['head aver '+str(options.averages['head_len'])+' pts'] + ['tailaver '+str(options.averages['tail_len'])+' pts'] + ['h-t difference'] + [fun['label'] for fun in options.functions]] \
                       + [[row.number] + [row.average_for_x_ms] + [row.head_average] +[row.tail_average] +[row.difference] + row.functions for row in self.chunks if not row.dropped] \
                       + [['average'] + [''] + ['' for fun in options.functions]+['']]
                       
            # result_1 = [['chunk\\average'] +  [['averages fo 10 ms'] +[fun['label'] for fun in options.functions]] \
            #            + [[row.number] +[row.average_for_x_ms]+  row.functions for row in self.chunks if not row.dropped] \
            #            + [['average'] +[''] + ['' for fun in options.functions]]
            
            
            #     mean_only = [self.mean_for_x_ms]
            #     mean_onlyT = numpy.transpose(mean_only)
        
            writer = csv.writer(csv_file, delimiter = ',')
            writer.writerows(result_1)
            
            print('Finished processing file ' + str(fn))
            print('\n Saved options:')
            print('head length (pts): '+ str(options.averages['head_len']))
            print('tail length (pts): '+ str(options.averages['tail_len']))
            s = str(options.averages['tail_head_diff'])
            print('Head-tail difference abs value:' , s)
            # print('Arithmetic mean filters: ' + + 'to' + str(options.functions['filter_max']))
            print('Arithmetic mean filters: -100 to  100')

    def plot_all(self, filename):
        from matplotlib.ticker import AutoMinorLocator
        import matplotlib.pyplot as plt

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        plt.title(str(fn)[:-9] +' All traces without filters')
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        plt.title(str(fn)[:-9] + ' All remaining traces (with filters)')
        for chunk in self.chunks:
            if not chunk.dropped:
                ax1.plot(chunk.timestamps, chunk.values, linewidth='0.8')
                ax2.plot(chunk.timestamps, chunk.corrected_values, linewidth='0.8')
                
        ax1.plot(self.chunks[0].timestamps, self.timestamps_averages, 'r', label='averages')
        ax2.plot(self.chunks[0].timestamps, self.timestamps_averages_corrected, 'r', label='averages')
        ax1.set_ylabel('Shift [pT]')
        ax2.set_ylabel('Shift [pT]')
        ax1.set_xlabel('Time [s]')
        ax2.set_xlabel('Time [s]')
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.grid(True, which='major')
        ax2.grid(True, which='major')
        ax1.grid(True, which='minor', linestyle=':')
        ax2.grid(True, which='minor', linestyle=':')
        
        # plt.title(str(fn)+' all shifted '+'drops '+ 'h-t '+ str(options.averages['tail_head_diff']))
        
        fig1.savefig(filename + '_all.png')
        fig2.savefig(filename + '_all_corrected.png')

    def plot_averages(self, filename):
        from matplotlib.ticker import AutoMinorLocator
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.chunks[0].timestamps, self.timestamps_averages_corrected, 'r', label='average')
        ax.set_ylabel('Shift [pT]')
        ax.set_xlabel('Time [s]')
        ax.legend(loc='best')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which='major')
        ax.grid(True, which='minor', linestyle=':')
        # plt.title(str(fn) + 'with baseline shifted'+' drops ' + 'h-t '+ str(options.averages['tail_head_diff'])+'arith mean -0.05 to 0.15 ')
        plt.title(str(fn[:-9]) +' Averaged traces not shifted')
        fig.savefig(filename + '_average.png')
        
    def plot_functions(self, filename):
        from matplotlib.ticker import AutoMinorLocator
        import matplotlib.pyplot as plt
        temp_chunks_no = [chunk.number for chunk in self.chunks if not chunk.dropped]
        functions = [[chunk.functions_corrected[idx] for chunk in self.chunks if not chunk.dropped]
                     for idx, function in enumerate(options.functions)]

        fig, ax = plt.subplots(figsize=(10, 6))
        for idx, function in enumerate(options.functions):
            ax.plot(temp_chunks_no, functions[idx], label=options.functions[idx]['label'])
        ax.set_ylabel('Shift [pT]')
        ax.set_xlabel('chunk number')
        ax.legend(loc='best')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which='major')
        ax.grid(True, which='minor', linestyle=':')
        fig.savefig(filename + '_functions.png')
        plt.title(str(fn)[:-9]+' Mean and std with applied filters')

    def plot_functions_nc(self, filename):
        from matplotlib.ticker import AutoMinorLocator
        import matplotlib.pyplot as plt

        temp_chunks_no = [chunk.number for chunk in self.chunks if not chunk.dropped]
        functions_nc = [[chunk.functions[idx] for chunk in self.chunks if not chunk.dropped]
                        for idx, function in enumerate(options.functions)]

        fig, ax = plt.subplots(figsize=(10, 6))
        for idx, function in enumerate(options.functions):
            ax.plot(temp_chunks_no, functions_nc[idx], label=options.functions[idx]['label'])
        ax.set_ylabel('Shift [pT]')
        ax.set_xlabel('chunk number')
        ax.legend(loc='best')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which='major')
        ax.grid(True, which='minor', linestyle=':')
        # plt.title(str(fn)+' dropped not corrected'' drops ' + 'h-t '+ str(options.averages['tail_head_diff'])+ 'arith mean -0.05 to 0.15')
        plt.title(str(fn)[:-9]+' Mean ans std not corrected')
        fig.savefig(filename + '_functions_nc.png')
        

def read_files():
    import csv
    import glob
    
    file_list = sorted(glob.glob(options.input_directory + str(fn)+'.csv'))
    print('Files to read: ', file_list)

    chunk = Chunk()
    data = Data()

    for file in file_list:
        print('Opening file: %s' % file)
        f = open(file, newline='')
        reader = csv.reader(f, delimiter=options.input_delimiter)
        if options.headers:
            next(reader)

        for row in reader:  # tutaj dodalam nowe kolumny
            chunk_no = int(row[0])
            timestamp = float(row[1])
            value = float(row[2])
            time_s = float(row[3])
            value_pt = float(row[4])
            
            error_deg = float(row[5])
            
            trigger = float(row[6])
            if not chunk.add_row(chunk_no, value, timestamp, time_s, value_pt, trigger, error_deg):
                data.add_chunk(chunk)
                chunk = Chunk()
                chunk.add_row(chunk_no, value, timestamp, time_s, value_pt, trigger, error_deg)

    data.add_chunk(chunk)
    print("Done reading file.")
    return data


def main():
    import sys

    try:
        data = read_files()
        data.compute_chunks()
        data.save_to_file(options.output_name)
        data.plot_all(options.output_name)
        data.plot_averages(options.output_name)
        data.plot_functions(options.output_name)
        data.plot_functions_nc(options.output_name)
    except (IOError, OSError) as e:
        print(e)
        sys.exit(e.errno)
    except Exception as e:
        print(e)
        sys.exit(-1)


if __name__ == "__main__":
    options = Options()
    main()
