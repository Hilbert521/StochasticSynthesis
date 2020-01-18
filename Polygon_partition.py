#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:36:21 2019

@author: apple
"""
#import matplotlib.pyplot as plt 

#region = [[[1,1], [1,2], [2,2], [2,1]],[[0,0], [4,0], [4,4], [1,4], [1,3], [0,3]]]
#print region

#x_axis = [[1,1,2,2],[0,4,4,1,1,0]]
#y_axis = [[1,2,2,1],[0,0,4,4,3,3]]

#plt.plot(x_axis, y_axis)
#plt.show()
#x1= [0,4,4,1,1,0]
#y1 = [0,0,4,4,3,3]
#plt.show(x1,y1)
#plt.show()


# =============================================================================
# import execjs
# k=execjs.eval("'red yellow blue'.split(' ')")
# ctx = execjs.compile("""
# 
#   function verifyDecomp(paths, ccw, expected) {
#     var rectangles = decomp(paths, ccw)
#     if(typeof expected !== "undefined") {
#       t.equals(rectangles.length, expected, "expected number of boxes")
#     }
#     t.same(boxOverlap(rectangles).filter(function(x) {
#       var a = rectangles[x[0]]
#       var b = rectangles[x[1]]
#       var x = Math.min(a[1][0], b[1][0]) - Math.max(a[0][0], b[0][0])
#       if(x <= 0) {
#         return false
#       }
#       var y = Math.min(a[1][1], b[1][1]) - Math.max(a[0][1], b[0][1])
#       if(y <= 0) {
#         return false
#       }
#       return true
#     }), [], "non-overlap")}
# 
#                    
#   function test(paths, ccw, expected) {
#     //Check all 4 orientations
#     for(var sx=1; sx>=-1; sx-=2)
#     for(var sy=1; sy>=-1; sy-=2)
#     {
#       var npaths = paths.map(function(path) {
#         return path.map(function(v) {
#           return [sx * v[0], sy * v[1]]
#         })
#       })
#       var nccw = sx * sy < 0 ? !ccw : ccw
#       verifyDecomp(npaths, nccw, expected)
#     }
#   }
# """)
# x= ctx.call("test",[
#       [[1,1], [1,2], [2,2], [2,1]],
#       [[0,0], [4,0], [4,4], [1,4], [1,3], [0,3]]
#     ], False, 4)
# print x
# =============================================================================


import os
from execjs import get
from matplotlib import pyplot as plt
from shapely.geometry.polygon import LinearRing, Polygon
from shapely.geometry import MultiPolygon

runtime = get()
ctx = runtime.compile('''
    module.paths.push('%s');
    var decompose = require('rectangle-decomposition'); 
    function decompose_region(region){
    var rectangles = decompose(region)
    return rectangles;
    }
    
''' % os.path.join(os.path.dirname(__file__),'node_modules'))

region = [[[1,1], [1,2], [2,2], [2,1]],[[0,0], [4,0], [4,4], [1,4], [1,3], [0,3]]]
x = ctx.call("decompose_region",region)
print "decompose_region 1", x

fig = plt.figure()
#poly = Polygon([(1,1), (1,2), (2,2), (2,1)])
poly = Polygon([(0,0),(2,0),(2,0.5),(1,0.5),(1,1),(0,1)])

x,y = poly.exterior.xy
ax = fig.add_subplot(111)
ax.plot(x, y, color='#6699cc', alpha=0.7,
    linewidth=3, solid_capstyle='round', zorder=2)
ax.set_title('Polygon')

region_2 = [[[0,0],[2,0],[2,0.5],[1,0.5],[1,1],[0,1]]]
x_2 = ctx.call("decompose_region",region_2)
print "decompose_region2 ", x_2


region_3 = [[[0,0],[2,0],[2,0.5],[1,0.5],[1,1],[0,1]],[[4,4],[5,4],[5,5],[4,5]]]
x_3 = ctx.call("decompose_region",region_3)
print "decompose_region 3 ", x_3

region_4 = [[[0.2, 0.19999999999999996], [0.2, 0.3875000000000002], [0.2150000000000003, 0.3875000000000002], [0.2150000000000003, 0.19999999999999996]]]
x_4 = ctx.call("decompose_region",region_4)
print "decompose_region 4 ", x_3

# =============================================================================
# import os
# import sys
# from execjs import get
# 
# runtime = get('Node')
# context = runtime.compile('''
#     module.paths.push('%s');
#     var ng = require('ng-annotate');
#     function annotate(src,cfg){
#         return ng(src,cfg);
#     }
# ''' % os.path.join(os.path.dirname(__file__),'node_modules'))
# 
# def ng_annotate(src,cfg=None):
#     if cfg is None:
#         cfg = dict(add=True)
#     return context.call('annotate',src,cfg)
# 
# def main():
#     print ng_annotate(open(sys.argv[-1],'r').read())
# 
# if __name__ == "__main__":
#     main()
# 
# =============================================================================

# =============================================================================
# from subprocess import call
# call(["node", "decomp.js"]) 
# =============================================================================




