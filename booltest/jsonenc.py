import json
import re


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


def unwrap(val):
    if isinstance(val, NoIndent):
        return unwrap_rec(val.value)
    return val


def unwrap_rec(val):
    if isinstance(val, list):
        return [unwrap_rec(x) for x in val]
    if isinstance(val, tuple):
        return tuple([unwrap_rec(x) for x in val])
    else:
        return unwrap(val)


class IndentingJSONEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

    def __init__(self, **kwargs):
        # Save copy of any keyword argument values needed for use here.
        self.__sort_keys = kwargs.get('sort_keys', None)
        super(IndentingJSONEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                else super(IndentingJSONEncoder, self).default(obj))

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.
        json_repr = super(IndentingJSONEncoder, self).encode(obj)  # Default JSON.

        try:
            from _ctypes import PyObj_FromPtr
        except:
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

