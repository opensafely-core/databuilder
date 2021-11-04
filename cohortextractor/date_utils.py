import calendar
import datetime


def cohort_date_range(start=None, end=None, increment="month"):
    if increment not in ["month", "week"]:
        raise ValueError(
            f"Unknown time period '{increment}': must be 'week' or 'month'"
        )
    if not (start or end):
        raise ValueError("At least one of start or end is required")
        # if only one of start/end date was provided, set both to the provided value
    start = start or end
    end = end or start

    start = _parse_date(start)
    end = _parse_date(end)
    if end < start:
        raise ValueError("Invalid date range: end cannot be earlier than start")
    dates = []
    while start <= end:
        dates.append(start.isoformat())
        start = _increment_date(start, increment)
    return dates


def _parse_date(date_str):
    if date_str == "today":
        return datetime.date.today()
    else:
        try:
            return datetime.date.fromisoformat(date_str)
        except ValueError:
            raise ValueError(f"Invalid date '{date_str}': Dates must be in YYYY-MM-DD")


def _increment_date(date, period):
    if period == "week":
        return date + datetime.timedelta(days=7)
    elif period == "month":
        if date.month < 12:
            try:
                return date.replace(month=date.month + 1)
            except ValueError:
                # If the month we've replaced the date in is out of range, it will be at the end
                # of a month which has fewer days than the previous month (e.g. 31st Aug + 1 month)
                # set to last day of previous month instead
                date = date.replace(day=1, month=date.month + 1)
                _, last_day_of_month = calendar.monthrange(date.year, date.month)
                return date.replace(day=last_day_of_month)
        else:
            return date.replace(month=1, year=date.year + 1)
    else:
        raise ValueError(f"Unknown time period '{period}'")