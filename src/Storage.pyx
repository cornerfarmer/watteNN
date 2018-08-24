import csv
from gym_watten.envs.watten_env cimport WattenEnv

cdef class Storage:

    def __init__(self, max_samples=0):
        self.max_samples = max_samples
        self.next_index = 0
        self.number_of_samples = 0
        self.key_numbers = {}
        if self.max_samples != 0:
            self.data.resize(max_samples)

    cdef int add_item(self, string key):
        pyKey = str(key)
        cdef int new_index
        if self.max_samples is 0:
            self.data.push_back(StorageItem())
            self.data.back().key = key
            if key != "":
                if pyKey in self.key_numbers:
                    self.key_numbers[pyKey] += 1
                else:
                    self.key_numbers[pyKey] = 1

            self.number_of_samples = self.data.size()
            return self.data.size() - 1
        else:
            new_index = self.next_index
            if self.number_of_samples > self.next_index and not self.data[self.next_index].value_net:
                self.key_numbers[str(self.data[self.next_index].key)] -= 1

            self.data[self.next_index] = StorageItem()
            self.data[self.next_index].key = key
            if pyKey != "":
                if pyKey in self.key_numbers:
                    self.key_numbers[pyKey] += 1
                else:
                    self.key_numbers[pyKey] = 1

            self.next_index += 1
            self.number_of_samples = max(self.number_of_samples, self.next_index)
            self.next_index %= self.max_samples
            return new_index

    cpdef void export_csv(self, file_name, WattenEnv env):
        with open(file_name, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            for i in range(self.number_of_samples):
                row = []
                for c in range(4):
                    for v in range(8):
                        if self.data[i].obs.sets[c][v][0] == 1:
                            row.append(env.filename_from_card(env.all_cards[c * 8 + v]).decode('utf-8'))

                for j in range(len(row), 3):
                    row.append('')

                for k in range(1, self.data[i].obs.sets[0][0].size()):
                    added = False
                    for c in range(4):
                        for v in range(8):
                            if self.data[i].obs.sets[c][v][k] == 1:
                                row.append(env.filename_from_card(env.all_cards[c * 8 + v]).decode('utf-8'))
                                added = True
                    if not added:
                        row.append('')

                row.append(self.data[i].obs.scalars[0])
                row.append(self.data[i].obs.scalars[2])

                for j in range(32):
                    if self.data[i].output.p[j] == 1:
                        row.append(env.filename_from_card(env.all_cards[j]).decode('utf-8'))

                row.append(self.data[i].output.v)
                row.append(self.data[i].weight)
                row.append(self.data[i].value_net)
                row.append(self.data[i].output.scale)

                writer.writerow(row)

    cdef void clear(self):
        self.data.clear()
        self.next_index = 0
        self.number_of_samples = 0
        if self.max_samples != 0:
            self.data.resize(self.max_samples)