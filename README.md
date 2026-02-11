### How to run Column Sketches?
First, you need to compile the project.

```sh
make release
```

Second, since column sketches currently do not support persistent storage, if you want to make column sketches effective, please create a new database file each time.

```duckdb
set threads to 1;
call dbgen(sf=10);
```

When the database connection is not closed, at this point, executing the query uses the column sketches.

```duckdb
pragma tpch(6);
```

Currently, only some columns of the lineitem table have undergone column sketches processing. For detailed make sketches, you can refer to src/storage/local_storage.cpp: void LocalStorage::Append(LocalAppendState &state, DataChunk &chunk). If you want to support other columns, you can make the modification.  

The specific processing logic for the sketch is located in src/include/duckdb/storage/statistics/column_sketch.hpp.

