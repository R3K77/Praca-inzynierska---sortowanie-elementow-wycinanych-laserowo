import json

json_data ={'sheet': {'bonusImages': {'MOG2_image': '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/wAALC...ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooor//2Q==', 'camera_image': '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDA...xerovntx0rqPgQi/8ACx9NXHBlGR+NFFd82+Q8PBpLGM6f/gpsqrrHgcBRgx3OePZa5X9ngBvhtKGGR9paiivSofw4ng8WJLOJekf/AElHSMi5PHeiiiug+csj/9k=', 'object_full_image': '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/wAALC...iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiv//Z'}, 'left_down_point': [1000, 480], 'right_down_point': [1345, 476], 'right_side_linear_fcn': [329, -1, -442029], 'right_up_point': [1344, 147], 'rotation': -0.1741507692065909, 'translation': (140.4253445494188, 80.95108097554731)}}
with open(f'cv_data_test_sheetTest.json', 'w', encoding='utf8') as f:
    json.dump(json_data, f, ensure_ascii=False)
