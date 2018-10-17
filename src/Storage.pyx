import csv
from gym_watten.envs.watten_env cimport WattenEnv

cdef class Storage:

    def __init__(self, max_samples=0):
        self.max_samples = max_samples
        self.next_index = 0
        self.number_of_samples = 0
        if self.max_samples != 0:
            self.data.resize(max_samples)

    cdef int add_item(self):
        cdef int new_index
        if self.max_samples is 0:
            self.data.push_back(StorageItem())

            self.number_of_samples = self.data.size()
            return self.data.size() - 1
        else:
            new_index = self.next_index
            self.data[self.next_index] = StorageItem()

            self.next_index += 1
            self.number_of_samples = max(self.number_of_samples, self.next_index)
            self.next_index %= self.max_samples
            return new_index


    cpdef void count_per_key(self, target_key):
        cdef float sum[32]
        for i in range(32):
            sum[i] = 0

        for i in range(self.number_of_samples):
            if not self.data[i].value_net:
                key = ""
                hand_cards_key = ""
                opponent_hand_cards_key = ""

                for k in range(self.data[i].obs.sets[0][0].size()):
                    for c in range(4):
                        for v in range(8):
                            if self.data[i].obs.sets[c][v][k] == 1:
                                if k == 0:
                                    hand_cards_key += str(c * 8 + v) + ","
                                elif k == 1 and self.data[i].value_net:
                                    opponent_hand_cards_key += str(c * 8 + v) + ","
                                elif k == 1 or (k == 2 and self.data[i].value_net):
                                    key = str(c * 8 + v) + ".1," + key
                                elif k >= 2:
                                    key = str(c * 8 + v) + "." + str(1 - self.data[i].obs.scalars[4 + k - (3 if self.data[i].value_net else 2)]) + "," + key
                key += "-" + hand_cards_key

                if target_key == key:
                    for j in range(32):
                        if self.data[i].output.p[j] == 1:
                            sum[j] += self.data[i].weight


        print(sum)


    cpdef void export_csv(self, file_name, WattenEnv env):
        with open(file_name, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            for i in range(self.number_of_samples):

                key = ""
                hand_cards_key = ""
                opponent_hand_cards_key = ""

                for k in range(self.data[i].obs.sets[0][0].size()):
                    for c in range(4):
                        for v in range(8):
                            if self.data[i].obs.sets[c][v][k] == 1:
                                if k == 0:
                                    hand_cards_key += str(c * 8 + v) + ","
                                elif k == 1 and self.data[i].value_net:
                                    opponent_hand_cards_key += str(c * 8 + v) + ","
                                elif k == 1 or (k == 2 and self.data[i].value_net):
                                    key = str(c * 8 + v) + ".1," + key
                                elif k >= 2:
                                    key = str(c * 8 + v) + "." + str(1 - self.data[i].obs.scalars[4 + k - (3 if self.data[i].value_net else 2)]) + "," + key
                key += "-" + hand_cards_key

                if self.data[i].value_net:
                    key += "-" + opponent_hand_cards_key


                row = []
                row.append(key)
                for c in range(4):
                    for v in range(8):
                        if self.data[i].obs.sets[c][v][0] == 1:
                            row.append(env.filename_from_card(env.all_cards[c * 8 + v]).decode('utf-8'))

                for j in range(len(row), 3):
                    row.append('')

                if self.data[i].value_net:
                    for c in range(4):
                        for v in range(8):
                            if self.data[i].obs.sets[c][v][1] == 1:
                                row.append(env.filename_from_card(env.all_cards[c * 8 + v]).decode('utf-8'))

                    for j in range(len(row), 6):
                        row.append('')

                for k in range(2 if self.data[i].value_net else 1, self.data[i].obs.sets[0][0].size()):
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

                if self.data[i].value_net:
                    row.append(self.data[i].output.v)
                else:
                    row.append(self.data[i].weight)
                row.append(self.data[i].value_net)

                writer.writerow(row)

    cdef void clear(self):
        self.data.clear()
        self.next_index = 0
        self.number_of_samples = 0
        if self.max_samples != 0:
            self.data.resize(self.max_samples)

    cdef void copy_from(self, Storage other_storage):
        cdef int i
        for i in range(other_storage.number_of_samples):
            storage_index = self.add_item()
            self.data[storage_index] = other_storage.data[i]