import itertools
import numpy as np
import bisect
from shapely import geometry


def fast_split(line, splitter):
    """
    Split a LineString with a Point or MultiPoint. 
    This function is a replacement for the shapely.ops.split function, but much faster.
    """

    if isinstance(splitter, geometry.Point):
        splitter = geometry.MultiPoint([splitter])

    # compute the distance from the beginning of the linestring for each point on line
    pts_on_line = list(
        itertools.compress(splitter, [line.distance(pt) < 1e-8 for pt in splitter])
    )
    splitter_distances = np.array([line.project(pt) for pt in pts_on_line])
    splitter_distances = splitter_distances[splitter_distances > 0]

    # compute accumulated distances from point-to-point on line of all
    # linestring coordinates
    ls_xy = np.array(line.xy).T
    ls_xy_roll = np.roll(ls_xy, 1, axis=0)
    eucl_dist = np.sqrt(
        (ls_xy_roll[:, 0] - ls_xy[:, 0]) ** 2 + (ls_xy_roll[:, 1] - ls_xy[:, 1]) ** 2
    )
    # the first distance is computed from the first point to the last point, set to 0
    eucl_dist[0] = 0
    ls_cumsum = eucl_dist.cumsum()

    # compute the indices on wich to split the line
    splitter_indices = np.unique(
        [bisect.bisect_left(ls_cumsum, splitter) for splitter in splitter_distances]
    ).astype(int)
    splitter_indices = splitter_indices[splitter_indices < (ls_xy.shape[0] - 1)]

    # split the linestring where each sub-array includes the split-point
    # create a new array with the index elements repeated
    tmp_indices = np.zeros(ls_xy.shape[0], dtype=int)
    tmp_indices[splitter_indices] = 1
    tmp_indices += 1
    ls_xy = np.repeat(ls_xy, tmp_indices, axis=0)

    # update indices to account for the changed array
    splitter_indices = splitter_indices + np.arange(1, len(splitter_indices) + 1)

    # split using the indices as usual
    slines = np.split(ls_xy, splitter_indices, axis=0)

    return slines


def junctions_shared_paths(coord_arr, idx_geom, shared_geoms, length_geoms):
    """
    collect junctions of shared paths
    
    Input
    -----
    coord_array : numpy.array
        array containing all linestrings 
    idx_geom : int
        slice index of linestring
    shared_geoms : list
        index values of linestrings that intersect linestring of idx_geom
    
    Returns
    -------
    junctions : numpy.array
        array containing all junctions from all intersecting linestrings
    """

    # currently fixed to solely 2D-coordinates
    no_dims = 2

    # get geoms that are rings
    first_xy_rows = coord_arr[shared_geoms, 0]
    last_xy_rows = coord_arr[:, length_geoms - 1][shared_geoms]
    last_xy_rows = last_xy_rows.diagonal(axis1=0, axis2=1).T
    rows_ring = (
        np.count_nonzero((first_xy_rows == last_xy_rows), axis=1) == no_dims
    ).nonzero()[0]

    # # for geoms that are rings, set first coord to np.nan
    # # since its equal to the last coord of rings
    # coord_arr[:, 0][rows_ring] = np.nan

    # create boolean of shared coords for each geom
    slice_array = (
        np.count_nonzero(np.isin(coord_arr[shared_geoms], coord_arr[idx_geom]), axis=2)
        == no_dims
    )

    # for geoms that are rings, set first coord to np.nan
    # since its equal to the last coord of rings
    slice_array[rows_ring, 0] = False

    # find edges of shared segments
    d = np.diff(slice_array)
    row, col, = d.nonzero()

    # set to float, so we can np.nan it
    row = row.astype(float)
    col = col.astype(float)

    # for rings we compare the 2nd coord (ix 1) with the last coord
    rings_start_shared = slice_array[:, 1]
    rings_end_shared = slice_array[:, length_geoms - 1]
    rings_end_shared = rings_end_shared.diagonal(axis1=0, axis2=1).T
    rings_start_end_shared = (rings_end_shared * rings_start_shared).nonzero()[0]

    col += 1

    # prepend a 0 for slice_array where start is True
    rows_first_true = np.isin(row, slice_array[:, 0].nonzero()[0])
    left_side_idx = np.unique(np.searchsorted(row, row, side="left"))
    insert_idx_left = left_side_idx[rows_first_true[left_side_idx]]

    row = np.insert(row, insert_idx_left, row[insert_idx_left])
    col = np.insert(col, insert_idx_left, 0)

    # append length of max-1 for end of slice_array is True
    rows_last_true = np.isin(row, slice_array[:, -1].nonzero()[0])
    right_side_idx = np.unique(np.searchsorted(row, row, side="right"))
    insert_idx_right = right_side_idx[rows_last_true[right_side_idx - 1]]

    row = np.insert(row, insert_idx_right, row[insert_idx_right - 1])
    col = np.insert(col, insert_idx_right, np.max(length_geoms) - 1)

    # from each segment subtract 1 from end idx, so we can slice in once
    col[1::2] -= 1

    # index of first element of each subsequence
    first_idx_first_segment_row = np.nonzero(np.r_[1, np.diff(row)[:-1]])[0]
    second_idx_last_segment_row = np.nonzero(np.r_[1, np.diff(row[::-1])[:-1]])[0]
    second_idx_last_segment_row = np.sort(
        ((len(row) - 1) - second_idx_last_segment_row)
    )

    # if shared path pass 0-index, set first and last junction to nan,
    # first idx of first segment to nan of shared paths passing 0-index
    row_bool = np.full(row.shape, False, dtype=bool)
    row_bool[first_idx_first_segment_row] = True
    first_idx_to_nan = row_bool * (row == rings_start_end_shared)

    row[first_idx_to_nan] = np.nan
    col[first_idx_to_nan] = np.nan

    # last idx of last segment to nan of shared paths passing 0-index
    row_bool = np.full(row.shape, False, dtype=bool)
    row_bool[second_idx_last_segment_row] = True
    last_idx_to_nan = row_bool * (row == rings_start_end_shared)

    row[last_idx_to_nan] = np.nan
    col[last_idx_to_nan] = np.nan

    # calculate exact index of junctions for take function
    col_idx = np.array((col * 2, col * 2 + 1)).T
    row_idx = row * (np.max(length_geoms)) * 2
    take_idx = col_idx + row_idx[None].T
    take_idx = take_idx[~np.isnan(take_idx).any(axis=1)].astype(int)

    if take_idx.size != 0:
        junctions = np.unique(coord_arr[shared_geoms].take(take_idx), axis=0)
    else:
        junctions = take_idx

    # \\ OLD 1 //
    # slice_array = np.isin(coord_arr[shared_geoms], coord_arr[idx_geom]).sum(axis=2) == 2

    # d = np.diff(slice_array)

    # # where geoms are rings and first and last coordinate are shared
    # # remove the junction in the end of the geom so a shared-path
    # # crossing the zero-index is preserved

    # # get geoms that are rings
    # first_xy_rows = coord_arr[:, 0, :]
    # last_xy_rows = coord_arr[:, length_geoms - 1, :]
    # last_xy_rows = last_xy_rows.diagonal(axis1=0, axis2=1).T
    # rows_ring = ((first_xy_rows == last_xy_rows).sum(axis=1) == 2).nonzero()[0]

    # # get rows where first and last coords are shared
    # rows_start_shared = slice_array[:, 0].nonzero()[0]
    # rows_end_shared = slice_array[
    #     range(slice_array.shape[0]), length_geoms - 1
    # ].nonzero()[0]

    # # get rings where first and last coords are shared
    # rings_start_shared = rows_start_shared[np.isin(rows_start_shared, rows_ring)]
    # rings_end_shared = rows_end_shared[np.isin(rows_end_shared, rows_ring)]
    # rings_start_end_shared = rows_end_shared[rows_end_shared == rings_start_shared]

    # # set last coord of rings to False where first and last coord is shared
    # length_geoms2 = length_geoms.copy()
    # length_geoms2[length_geoms == length_geoms.max()] -= 1
    # length_geoms2 = length_geoms2[rings_start_end_shared]
    # diag_to_false = d[rings_start_end_shared[None].T, length_geoms2 - 1]

    # np.fill_diagonal(diag_to_false, False)
    # d[rings_start_end_shared[None].T, length_geoms2 - 1] = diag_to_false

    # row, col, = d.nonzero()

    # col += 1
    # # geoms were last coordinates end with True should not get +1, do -1
    # rows_xy_end_true = np.isin(row, rows_end_shared)
    # col[rows_xy_end_true] -= 1

    # # every odd index is an end of segment, subtract 1
    # col[1::2] -= 1

    # # calculate exact index of junctions for take function
    # col_idx = np.array((col * 2, col * 2 + 1)).T
    # row_idx = row * (np.max(length_geoms)) * 2
    # take_idx = col_idx + row_idx[None].T

    # junctions = np.unique(coord_arr.take(take_idx), axis=0)
    # junctions = junctions[np.isin(junctions, coord_arr[idx_geom])[:, 0]]

    # \\ OLD 2 //
    # d = np.diff(slice_array)
    # row, col, = d.nonzero()

    # col += 1
    # # geoms were last coordinates end with True should not get +1, do -1
    # rows_xy_end_true = np.isin(
    #     row, slice_array[range(slice_array.shape[0]), length_geoms - 1].nonzero()[0]
    # )
    # col[rows_xy_end_true] -= 1

    # # prepend a 0 for slice_array where start is True
    # rows_first_true = np.isin(row, slice_array[:, 0].nonzero()[0])
    # left_side_idx = np.unique(np.searchsorted(row, row, side="left"))
    # insert_idx_left = left_side_idx[rows_first_true[left_side_idx]]

    # row = np.insert(row, insert_idx_left, row[insert_idx_left])
    # col = np.insert(col, insert_idx_left, 0)

    # # append length of max-1 for end of slice_array is True
    # rows_last_true = np.isin(row, slice_array[:, -1].nonzero()[0])
    # right_side_idx = np.unique(np.searchsorted(row, row, side="right"))
    # insert_idx_right = right_side_idx[rows_last_true[right_side_idx - 1]]

    # row = np.insert(row, insert_idx_right, row[insert_idx_right - 1])
    # col = np.insert(col, insert_idx_right, np.max(length_geoms) - 1)

    # # calculate exact index of junctions for take function
    # col_idx = np.array((col * 2, col * 2 + 1)).T
    # row_idx = row * (np.max(length_geoms)) * 2
    # take_idx = col_idx + row_idx[None].T

    # # collect coordinates of junctions
    # junctions = coord_arr.take(take_idx)

    return junctions


def insertor(geoms):
    """
    generator function to use stream loading of geometries for creating a rtree index
    """

    for i, obj in enumerate(geoms):
        yield (i, obj.bounds, None)


def get_matches(geoms, tree_idx):
    """
    Function to return the indici of the rtree that intersects with the input geometries
    
    Parameters
    ----------
    geoms : list
        list of geometries to compare against the rtree index
    tree_idx: rtree.index.Index
        an rtree indexing object
        
    Returns
    -------
    matches: list
        list of tuples, where the key of each tuple is the linestring index 
        and the value of each key is a list of junctions intersecting bounds of linestring
    """

    matches = []
    for idx_ls, obj in enumerate(geoms):
        intersect_idx = list(tree_idx.intersection(obj.bounds))
        if len(intersect_idx):
            # matches.append([[idx_ls], intersect_idx])
            intersect_idx.sort()
            matches.append([idx_ls, intersect_idx])
    return matches


def select_unique(data):
    sorted_data = data[np.lexsort(data.T), :]
    row_mask = np.append([True], np.any(np.diff(sorted_data, axis=0), 1))

    return sorted_data[row_mask]


def select_unique_combs(linestrings):
    try:
        from rtree import index
    except:
        raise "without rdtree not implemented"
        # all_line_combs = list(itertools.combinations(range(len(linestrings)), 2))
        # return all_line_combs
    # create spatial index on junctions including performance properties
    p = index.Property()
    p.leaf_capacity = 1000
    p.fill_factor = 0.9
    tree_idx = index.Index(insertor(linestrings), properties=p)

    # get index of linestrings intersecting each linestring
    idx_match = get_matches(linestrings, tree_idx)

    # # make combinations of unique possibilities
    # combs = []
    # for idx_comb in idx_match:
    #     combs.extend(list(itertools.product(*idx_comb)))

    # combs = np.array(combs)
    # combs.sort(axis=1)
    # combs = select_unique(combs)

    # uniq_line_combs = combs[(np.diff(combs, axis=1) != 0).flatten()]

    return idx_match  # uniq_line_combs
