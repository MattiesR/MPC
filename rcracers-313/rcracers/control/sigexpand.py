from contextlib import suppress
import textwrap
from inspect import signature, cleandoc
from types import FunctionType
from typing import Callable, List


def _process_expand(f, template):
    validate(f, template)
    p = signature(f).parameters
    ps = signature(template).parameters
    if set(p.keys()) == set(ps.keys()):
        return f

    def wrap(*args, **kwargs):
        res = {}
        for i, k in enumerate(ps):
            if k in p:
                with suppress(IndexError):
                    res[k] = args[i]
                    continue
                res[k] = kwargs[k]
        return f(**res)

    return wrap


def expand(f: Callable = None, /, *, template: Callable):
    """Expand the signature of the provided callable.

    Args:
        template (Callable): The template to use during expansion
        f (Callable, optional): The callable to expand. Defaults to None.
    """ 
    def wrap(f):
        return _process_expand(f, template)

    # See if we're being called as @expand or @expand().
    if f is None:
        # We're called with parens.
        return wrap

    # We're called as @expand without parens.
    return wrap(f)


def validate(f: Callable, template: Callable):
    """Validate whether a callable satisfies the provided template

    Args:
        f (Callable): The callable to check
        template (Callable): The template which the callable should satisfy

    Raises:
        SignatureException: The callable does not satisfy the template.
    """
    sig = signature(template)
    ps = sig.parameters
    p = signature(f).parameters
    missing = []
    for k in p:
        if k not in ps:
            missing.append(k)
    if len(missing) > 0:
        if isinstance(f, FunctionType):
            name = f.__name__
        else:
            name = f.__class__.__name__
        raise SignatureException(name, str(sig), missing, template.__doc__)


class SignatureException(Exception):
    """Signature Exception raised when a callable does not satisfy a template."""
    
    def __init__(self, name: str, sig: str, missing: List[str], doc: str = None):
        self.f_name = name
        self.sig = sig
        self.missing = missing
        self.doc = doc
        super().__init__(
            f"Method {self.f_name} has arguments not specified by signature."
        )

    def __str__(self):
        res = str().join(
            (
                f'\nMethod "{self.f_name}" has arguments not specified by signature:\n',
                f"  Supported (maximal) signature:\n"
                f"    {str(self.sig)}\n"
                f"  Unsupported arguments:\n",
                f"    {str(tuple(self.missing))[:-2]})",
            )
        )
        if self.doc is not None:
            doc = textwrap.indent(
                '"""\n' + cleandoc(self.doc) + '\n"""', prefix=" " * 2
            )
            res += "\nSignature Documentation:\n" + doc
        return textwrap.indent(res, prefix=" " * 2)
