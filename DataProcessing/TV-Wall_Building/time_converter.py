
def video_time_str_to_seconds(video_time: str):
    """
    Convert a video time's string representation into numeric value in seconds
    :param video_time: in a format "hh:mm:ss" or "mm:ss"
    :return: int - seconds from start of the video
    """
    secs = 0
    factor = 1
    items = video_time.split(':')
    items.reverse()
    for t in items:
        digit = 0
        try:
            digit = int(t.lstrip("0"))
        except ValueError:
            pass
        secs += factor * digit
        factor *= 60
    return secs


def video_time_to_index(video_time: str, sample_rate: float):
    secs = video_time_str_to_seconds(video_time)
    return int(secs * sample_rate)


if __name__ == '__main__':
    print(video_time_to_index("01:00:03"))