# chunk sizes bigger than this value can trigger a compression error
# https://github.com/scverse/spatialdata/issues/812#issuecomment-2559380276
# so if we detect this during parsing/validation we raise a warning
MAX_N_ELEMS_CHUNK_SIZE = 2147483647
