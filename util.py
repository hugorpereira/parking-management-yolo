def shift_points_horizontal(data, shift):
    for item in data:
        new_points = []
        for x, y in item["points"]:
            new_points.append([x + shift, y])
        item["points"] = new_points
    return data

def check_offset(frame_count, original_boxes):

    frame_slice_offset = [
        [10, 50, 3],
        [50, 80, 7],
        [80, 200, 5],
        [200, 300, -8],
        [300, 370, -9],
        [370, 400, -9],
        [400, 500, -9],
        [500, 600, -9],
    ]

    slice_item = next(
        (s for s in frame_slice_offset if s[0] <= frame_count < s[1]),
        None
    )

    if slice_item is None:
        return []

    start, end, shift = slice_item
    px_per_frame = shift / (end - start)
    return shift_points_horizontal(original_boxes, px_per_frame)