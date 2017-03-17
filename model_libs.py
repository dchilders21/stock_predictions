"""
Convert a datetime object from one timezone to a new timezone
Usage:
from tz2ntz import tz2ntz
import datetime
tz2ntz(datetime.datetime.utcnow(), 'UTC', 'US/Pacific')
datetime.datetime(2015, 8, 5, 1, 35, 8, 791928, tzinfo=<DstTzInfo 'US/Pacific' PDT-1 day, 17:00:00 DST>)
print tz2ntz(datetime.datetime.utcnow(), 'UTC', 'US/Pacific')
2015-08-05 01:39:43.625592-07:00
"""
import datetime
import pytz


def tz2ntz(date_obj, tz, ntz):
    """
    :param date_obj: datetime object
    :param tz: old timezone
    :param ntz: new timezone
    """
    if isinstance(date_obj, datetime.date) and tz and ntz:
       date_obj = date_obj.replace(tzinfo=pytz.timezone(tz))
       return date_obj.astimezone(pytz.timezone(ntz))
    return False