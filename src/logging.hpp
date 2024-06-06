#pragma once

void logger_log(
    int level,
    const char * f_name,
    int line,
    const char * str,
    ...
);

#define DEBUG_LOG( ... ) logger_log( 0, __func__, __LINE__, __VA_ARGS__ );
#define INFO_LOG( ... )  logger_log( 1, __func__, __LINE__, __VA_ARGS__ );
#define ERROR_LOG( ... ) logger_log( 2, __func__, __LINE__, __VA_ARGS__ );
