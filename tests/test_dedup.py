import unittest
import topojson    

class TestDedup(unittest.TestCase): 
    # duplicate rotated geometry bar with hole interior in geometry foo
    def test_duplicate_rotated_hole_interior(self): 
        data = {
            "foo": {
                "type": "MultiPolygon",
                "coordinates": [
                    [
                        [[0, 0], [20, 0], [10, 20], [0, 0]], # CCW
                        [[3, 2], [10, 16], [17, 2], [3, 2]] # CW
                    ],
                    [
                        [[6, 4], [14, 4], [10, 12], [6, 4]] #CCW 
                    ]

                ]
            },
            "bar": {"type": "Polygon", "coordinates": [[[17, 2], [3, 2], [10, 16], [17, 2]]]}
        }      
        topo = topojson.dedup(topojson.cut(topojson.join(topojson.extract(data))))
        # print(topo)
        self.assertEqual(len(topo['duplicates']), 0)
        self.assertEqual(topo['bookkeeping_geoms'], [[0, 2], [1], [2]])

       
