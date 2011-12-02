# http://code.activestate.com/recipes/117228/

class PriorityDictionary(dict):
    """
    this data structure acts almost like a dictionary, with two modifications:
    first, D.smallest() returns the value x minimizing D[x].
    second, iterating "for x in D" finds and removes the items from D in sorted order.
    """
    
    def __init__(self):
        self.__heap = []
        dict.__init__(self)

    def smallest(self):
        """ finds smallest item after removing deleted items from heap. """
        if len(self) == 0:
            raise IndexError, "Smallest of empty PriorityDictionary"
        heap = self.__heap
        while heap[0][1] not in self or self[heap[0][1]] != heap[0][0]:
            lastItem = heap.pop()
            insertionPoint = 0
            while 1:
                smallChild = 2 * insertionPoint + 1
                if smallChild + 1 < len(heap) and heap[smallChild] > heap[smallChild+1]:
                    smallChild += 1
                if smallChild >= len(heap) or lastItem <= heap[smallChild]:
                    heap[insertionPoint] = lastItem
                    break
                heap[insertionPoint] = heap[smallChild]
                insertionPoint = smallChild
        return heap[0][1]

    def __iter__(self):
        """ creates destructive sorted iterator of PriorityDictionary. """
        def iterfn():
            while len(self) > 0:
                x = self.smallest()
                yield x
                del self[x]
        return iterfn()

    def __setitem__(self, key, val):
        """ 
        changes value stored in dictionary and add corresponding
        pair to heap. rebuilds the heap if the number of deleted
        items grows too large, to avoid memory leakage.
        """
        dict.__setitem__(self, key, val)
        heap = self.__heap
        if len(heap) > 2 * len(self):
            self.__heap = [(v, k) for k, v in self.iteritems()]
            self.__heap.sort()
        else:
            newPair = (val, key)
            insertionPoint = len(heap)
            heap.append(None)
            while insertionPoint > 0 and newPair < heap[(insertionPoint-1)//2]:
                heap[insertionPoint] = heap[(insertionPoint-1)//2]
                insertionPoint = (insertionPoint-1)//2
            heap[insertionPoint] = newPair

    def setdefault(self, key, val):
        """ reimplements setdefault to call customized __setitem__. """
        if key not in self:
            self[key] = val
        return self[key]
