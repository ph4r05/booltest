import json
import re
import collections


class NoIndent(object):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return repr(self.value)

    def __eq__(self, other):
        return self.value.__eq__(other)

    def __cmp__(self, other):
        return self.value.__cmp__(other)

    def __hash__(self):
        return self.value.__hash__()


def unwrap_rec(val, rec_fnc=None):
    rec_fnc = rec_fnc if rec_fnc is not None else unwrap_rec
    if isinstance(val, list):
        return [rec_fnc(x) for x in val]
    elif isinstance(val, tuple):
        return tuple([rec_fnc(x) for x in val])
    elif isinstance(val, collections.OrderedDict):
        return collections.OrderedDict([(x, rec_fnc(val[x])) for x in val])
    elif isinstance(val, dict):
        return dict([(x, rec_fnc(val[x])) for x in val])
    elif isinstance(val, NoIndent):
        return rec_fnc(val.value)
    else:
        return val


class IndentingJSONEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

    def __init__(self, **kwargs):
        # Save copy of any keyword argument values needed for use here.
        self.__sort_keys = kwargs.get('sort_keys', None)
        self.__unwrapper = kwargs.get('unwrapper', unwrap_rec)
        super(IndentingJSONEncoder, self).__init__(**kwargs)

    def unwrap(self, obj):
        if self.__unwrapper:
            obj = self.__unwrapper(obj)
        return obj

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                else super(IndentingJSONEncoder, self).default(obj))

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.
        has_ctypes = False
        try:
            from _ctypes import PyObj_FromPtr
            has_ctypes = True

        except:
            obj = self.unwrap(obj)

        json_repr = super(IndentingJSONEncoder, self).encode(obj)  # Default JSON.
        if not has_ctypes:
            return json_repr

        # Replace any marked-up object ids in the JSON repr with the
        # value returned from the json.dumps() of the corresponding
        # wrapped Python object.
        for match in self.regex.finditer(json_repr):
            # see https://stackoverflow.com/a/15012814/355230
            id = int(match.group(1))
            no_indent = PyObj_FromPtr(id)
            json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)

            # Replace the matched id string with json formatted representation
            # of the corresponding Python object.
            json_repr = json_repr.replace(
                            '"{}"'.format(format_spec.format(id)), json_obj_repr)

        return json_repr

