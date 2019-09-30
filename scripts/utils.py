import datetime

def get_time_stamp():
    """Returns current time stamp
    """

    time_stamp = str(datetime.datetime.now()).split('.')[0]
    time_stamp = time_stamp.replace(' ', '_').replace(':', '-')
    return time_stamp

def get_formatted_time(t):
    """Formats time t to string
    """

    if t < 60:
        return "%.1fs" % (t)
    elif t < 3600:
        return "%.1fm" % (t/60)
    else:
        return "%.1fh" % (t/3600)
