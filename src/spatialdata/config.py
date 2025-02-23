# chunk sizes bigger than this value (bytes) can trigger a compression error
# https://github.com/scverse/spatialdata/issues/812#issuecomment-2559380276
# so if we detect this during parsing/validation we raise a warning
LARGE_CHUNK_THRESHOLD_BYTES = 2147483647
