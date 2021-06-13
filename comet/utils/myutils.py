def arg2dict(arg):
    # converts a string like opt1:val, opt2:val to a dictionary
    return dict(
        map(str.strip, sub.split(":", 1))
        for sub in arg.split(",")
        if ":" in sub
    )
