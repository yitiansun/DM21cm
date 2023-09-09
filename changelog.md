# Some principles:
- A function should only do one thing, and exactly what the name implies.
- A function/wrapper should not know about things that it does not do. (E.g. if we do synchronization outside of dh_wrapper, dh_wrapper should not know about synchronization)