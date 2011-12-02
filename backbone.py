#!/opt/python/bin/python

import timeit
import numpy
import math
import sys
import matplotlib.pyplot as plot
from prioritydict import PriorityDictionary

# shortest path table, its structure is this:
# key: (clst1, clst2), clst1 and clst2 are the label of clusters
# the order doesn't matter, because forward and backward are the same
# value: shortest path instance
SP_TABLE = {}

# final network graph, it is created in order to calculate
# shortest path between two given points
GRAPH = {}

class Point(object):
    """
    represents a point in the NxN plane.
    """

    def __init__(self, id = 0, x = 0.0, y = 0.0):
        """
        constructor
        id is the identification of current point, if the point is used to represent centroid, id is always 0
        x indicates x coordinate
        y indicates y coordinate
        """
        self.id = id
        self.x = x
        self.y = y

    def distance(self, pt):
        """ computes euclidean distance between two points. """
        return math.sqrt((self.x - pt.x) ** 2 + (self.y - pt.y) ** 2)

    def frequency(self, pt):
        """ computes frequency between two points. """
        return math.floor(abs(self.id - pt.id) / 2)

    def __eq__(self, pt):
        """ whether two points are the same. """
        return self.x == pt.x and self.y == pt.y

    def __repr__(self):
        """ used for debugging. """
        return "Point " + str(self.id) + " (" + str(self.x) + ", " + str(self.y) + ")"

class Cluster(object):
    """ it represents a cluster/classification which has classified points. """
    def __init__(self, centroid):
        """
        constructor
        label means the identification of current cluster
        centroid is the centroid point of current cluster
        points means all the points which are belong to current cluster
        radius is used to draw plot.
        """
        self.label = ''
        self.centroid = centroid
        self.points = []
        self.radius = 0.0   # used to draw plot
        self.neighbour = {}
        self.inter_cost = 0
        self.intra_cost = 0
        self.dm_cost = 0

    def add(self, pt):
        """ adds new point to the cluster. """
        self.points.append(pt)

    def calc_centroid(self):
        """ calculates new centroid based on classified points. """
        sumX = 0.0
        sumY = 0.0
        dis = 0.0
        for p in self.points:
            sumX += p.x
            sumY += p.y
            d = p.distance(self.centroid)
            if dis < d: dis = d
        # radius is the longest distance within points
        self.radius = dis + 0.1
        size = len(self.points)
        if size:
            return Point(x=float(sumX)/size, y=float(sumY)/size)
        else:
            return self.centroid

    def update(self):
        """ updates new centroid and calculates diff between old and new centroid. """
        old_centroid = self.centroid
        self.centroid = self.calc_centroid()
        return old_centroid.distance(self.centroid)

    def __eq__(self, clst):
        """ whether two clusters are the same. """
        return self.centroid == clst.centroid

    def __repr__(self):
        """ used for debugging. """
        return "cluster '" + self.label + "' at " + str(self.centroid)

def create_points(N, M):
    """
    create NxN plane with M points randomly distributed.
    N: domain size of plane
    M: number of points
    """
    arr = numpy.random.randint(0, N+1, size=(M, 2))
    idx = 0
    coords = []
    points = []
    # for ele in arr:
    #     pt = Point(x = ele[0], y = ele[1])
    #     if pt not in points:
    #         idx += 1
    #         pt.id = idx
    #         points.append(pt)

    # if previous step creates duplicated points,
    # we have to make sure create more points.
    # while idx < M:
    #     print "Am I possible got called?"
    #     missed = numpy.random.randint(0, N+1, size=(1, 2))
    #     pt = Point(x = missed[0][0], y = missed[0][1])
    #     if pt not in points:
    #         idx += 1
    #         pt.id = idx
    #         points.append(pt)
    
    for ele in arr:
        if (ele[0], ele[1]) not in coords:
            idx += 1
            coords.append((ele[0], ele[1]))
    
    while idx < M:
        missed = numpy.random.randint(0, N+1, size=(M-idx, 2))
        for ele in missed:
            if (ele[0], ele[1]) not in coords:
                idx += 1
                coords.append((ele[0], ele[1]))
    idx = 0
    for coord in coords:
        idx += 1
        points.append(Point(id=idx, x=coord[0], y=coord[1]))

    return points

def create_clusters(N, K):
    """ 
    create randomly distributed k clusters.
    N: domain size of plane
    K: number of clusters
    """
    clusters = []
    centroids = create_points(N, K)
    for idx, centroid in enumerate(centroids):
        cluster = Cluster(centroid)
        cluster.label = chr(96+idx+1)
        clusters.append(cluster)
    return clusters

def _empty_clusters(clusters):
    """ removes all the points in the cluster. """
    for clst in clusters:
        clst.points = []

def kmeans(points, clusters, threshold=1e-10):
    """
    1. put each point into nearest cluster
    2. calculate new centroid for each cluster
    3. repeat step 2 and step 3 until diff is less than threshold
    """
    diff = threshold + 1.0
    while diff > threshold:
        _empty_clusters(clusters)
        for pt in points:
            min_dis = sys.maxint
            min_cls = None
            for clst in clusters:
                dis = pt.distance(clst.centroid)
                if min_dis > dis:
                    min_dis = dis
                    min_cls = clst
            min_cls.add(pt)
        diff = 0.0

        for clst in clusters:
            diff += clst.update()

def create_backbone_network(G, clusters, distance):
    """ 
    creates the backbone network with given clusters.
    G: the final graph
    clusters: stable k clusters
    distance: if distance between two centroids are less or equal to distance,
              should be connected
    """
    G.clear()   # clear the graph before assigning new components
    for clst in clusters:
        neighbour = {}
        for peer in clusters:
            dis = clst.centroid.distance(peer.centroid)
            if clst != peer and dis <= distance:
                neighbour[peer.label] = dis
                clst.neighbour[peer] = dis
        G[clst.label] = neighbour

def find_cluster(clusters, label):
    """ finds the exact cluster by given label"""
    for clst in clusters:
        if clst.label == label: return clst
    return None

def _dijkstra(G, start, end=None):
    """ Dijkstra's shortest path algorithm. """
    distances = {}
    predecessors = {}
    Q = PriorityDictionary()
    Q[start] = 0

    for pt in Q:
        distances[pt] = Q[pt]
        if pt == end: break

        for neigh in G[pt]:
            pnLength = distances[pt] + G[pt][neigh]
            if neigh in distances:
                if pnLength < distances[neigh]:
                    raise ValueError, "Dijkstra: found better path to already-final vertex"
            elif neigh not in Q or pnLength < Q[neigh]:
                Q[neigh] = pnLength
                predecessors[neigh] = pt
    return distances, predecessors

def shortest_path(G, start, end):
    """
    finds a single shortest path from given start point to the given end point.
    first check whether it is in the cached shortest path table (SP_TABLE),
    if it has been calculated before, directly pick it there;
    otherwise calculate the new shortest path and store it to the cache table.
    """
    if (start, end) in SP_TABLE:
        return SP_TABLE[(start, end)]
    elif (end, start) in SP_TABLE:
        return SP_TABLE[(end, start)]
    else:
        D, P = _dijkstra(G, start, end)
        path = []
        temp = end
        while 1:
            path.append(end)
            if end == start: break
            end = P[end]
        path.reverse()
        SP_TABLE[(start, temp)] = path
        return path
    
def find_all_shortest_paths(clusters, cached):
    """ finds all shortest paths and stores them in the shortest path table. """
    cached.clear()    # clear cached table for new size of clusters
    for clst1 in clusters:
        for clst2 in clusters:
            if clst1 != clst2:
                shortest_path(GRAPH, clst1.label, clst2.label)

def inter_cost(cluster):
    """
    calculates the inter network cost.
    each two points in the same cluster will be connected
    by the centroid of the cluster.
    cluster: given cluster
    """
    inter_sum = 0
    for src in cluster.points:
        for dest in cluster.points:
            if src != dest:
                inter_sum += src.frequency(dest)
    return int(inter_sum)    # because (a, b) and (b, a) is the same

def intra_cost(clusters, cluster):
    """
    calculates the intra network cost.
    each two points belong to different clusters.
    clusters: the whole network
    cluster: given cluster
    """
    intra_sum = 0
    for src in cluster.points:
        for clst in clusters:
            if clst != cluster:
                for dest in clst.points:
                    intra_sum += src.frequency(dest)
    return int(intra_sum/2)
    
def _door_matt(sclst, eclst):
    """
    calculates the communication frequency between two clusters
    sclst: source cluster
    eclst: end cluster
    """
    dm_sum = 0
    for src in sclst.points:
        for dest in eclst.points:
            dm_sum += src.frequency(dest)
    return int(dm_sum)

def door_matt_cost(clusters, cluster):
    """
    calculates the door matt effect cost.
    cluster: given cluster which participates as a stepping stone
    in a communication of points.
    """
    # checks shortest path table, finds all path larger than 3,
    # then the cluster between will be used as door matt.
    dm_sum = 0
    for path in SP_TABLE.values():
        if (len(path) > 2) and (cluster.label in path[1:-1]):
            src = find_cluster(clusters, path[0])
            dest = find_cluster(clusters, path[-1])
            dm_sum += _door_matt(src, dest)
    return int(dm_sum)

# def _shortest(start, end, path, visited):
#     if start == end:
#         path[:] = [end]
#         return 0
#     
#     currentBestLen = sys.maxint
#     currentBestRoute = []
#     visited.append(start)
#     for neigh in start.connections:
#         if neigh in visited: continue
#         # visited.append(neigh)
#         aiResult = []
#         cost = _shortest(neigh, end, aiResult, visited)
#         # visited.remove(neigh)
#         if cost + start.connections[neigh] < currentBestLen:
#             currentBestLen = cost + start.connections[neigh]
#             currentBestRoute = [start] + aiResult
#     path[:] = currentBestRoute
#     visited.remove(start)
#     return currentBestLen

def draw_plot(points, clusters):
    plot.title("k-means back bone network")
    plot.axis([-1, 26, -1, 26])
    plot.xticks(range(-1, 26, 1))
    plot.yticks(range(-1, 26, 1))
    # plot.plot([pt.x for pt in points], [pt.y for pt in points], 'o')
    for pt in points:
        plot.text(pt.x+0.1, pt.y+0.1, str(pt.id))
    for clst in clusters:
        color = (numpy.random.rand(), numpy.random.rand(), numpy.random.rand())
        cir = plot.Circle((clst.centroid.x, clst.centroid.y), radius=clst.radius, alpha=0.3, fc=color)
        plot.gca().add_patch(cir)
        plot.plot([clst.centroid.x], [clst.centroid.y], '^')
        plot.plot([pt.x for pt in clst.points], [pt.y for pt in clst.points], 'o')

        draw_connections(plot, GRAPH, clusters)
    plot.grid(True)
    plot.show()
    
def draw_connections(plot, graph, clusters):
    for start in graph:
        sc = find_cluster(clusters, start)
        for ec in sc.neighbour:
            plot.plot([sc.centroid.x, ec.centroid.x], [sc.centroid.y, ec.centroid.y], '-k', linewidth=0.1)
            
def total_cost(clusters):
    inter = 0
    intra = 0
    dm = 0
    total = 0
    for clst in clusters:
        print clst.label, "has cost: ", str(clst.inter_cost), str(clst.intra_cost), str(clst.dm_cost)
        inter += clst.inter_cost
        intra += clst.intra_cost
        dm += clst.dm_cost
        # total += (clst.inter_cost + clst.intra_cost + clst.dm_cost)
    total = inter + intra + dm
    return (inter, intra, dm, total)
    # return total

def draw_cost_plot(hori, verts):
    plot.title('find cost')
    plot.xticks(range(-1, 40, 1))
    
    _draw_line(plot, hori, verts[0], 'g')
    _draw_line(plot, hori, verts[1], 'b')
    _draw_line(plot, hori, verts[2], 'r')
    _draw_line(plot, hori, verts[3], 'k')
    # plot.plot(hori, vert, '-o')
    
    plot.grid(True)
    plot.show()
    
def _draw_line(plot, hori, vert, color):
    plot.plot(hori, vert, '-o'+color)
    
DEBUG = False

def main():
    Ks = None
    if DEBUG:
        Ks = [10]
    else:
        Ks = [3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 30]
    
    iec = []
    iac = []
    dmc = []
    costs = []
    
    points = create_points(25, 100)

    clusters = None
    
    for k in Ks:
        print "===================", k, "=========================="
        clusters = create_clusters(25, k)

        kmeans(points, clusters)
#        print "Finished creating kmeans algorithm"

        create_backbone_network(GRAPH, clusters, math.sqrt(3)*25/2.0)
#        print "Finished creating backbone network"

        find_all_shortest_paths(clusters, SP_TABLE)
#        print "Finished finding all shortest paths"
    
        for clst in clusters:
            clst.inter_cost = inter_cost(clst)
            clst.intra_cost = intra_cost(clusters, clst)
            clst.dm_cost = door_matt_cost(clusters, clst)

        ret = total_cost(clusters)
        iec.append(ret[0])
        iac.append(ret[1])
        dmc.append(ret[2])
        costs.append(ret[3])
        # costs.append(total_cost(clusters))
    draw_cost_plot(Ks, [iec, iac, dmc, costs])
    # draw_plot(points, clusters)

if __name__ == '__main__':
    if DEBUG:
        t = timeit.Timer("main()", "from __main__ import main")
        print t.timeit(5)
    else:
        main()
