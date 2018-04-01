
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