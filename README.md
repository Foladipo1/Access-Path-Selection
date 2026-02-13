### About this project

This project contains a framework for students to quickly configure and evaluate access path selection strategies in DuckDB, a vectorized push-based columnar DBMS. 

The code repos includes a sophisticated implementation developed by the TAs, allowing students to benchmark it against native zone maps and bitmap indexing quickly.

### Read the code

Please read the following code snippets:

- src/include/duckdb/storage/statistics/column_sketch.hpp contains the core Column Sketches implementation. Our implementation leverages AVX-512 acceleration and requires a modern Intel or AMD CPU with AVX-512 support. 

- planner/filter/*_filter.cpp:CheckSketchStatistics() implements the logic for data skipping based on sketches of column data. Note that the neighboring CheckStatistics() function handles the equivalent logic for the native zone maps.

- storage/statistics/numeric_stats.cpp:CheckSketchTemplated() defines how column sketches are evaluated against specific predicates. E.g., "A = x" and  "A <= x".

### Run the code

First, compile the project.

```sh
make release
```

Second, generate workloads and column sketches for columns involved in TPC-H Q6. This step takes a few minutes, (mainly) depending on your CPU HZ number.

```duckdb
set threads to 1;
call dbgen(sf=10);
```

Run the TPC-H Q6 using column sketches.

```duckdb
pragma tpch(6);
```

Note that the above `dbgen` command generates column sketches for columns involved in TPC-H Q6 automatically. If you want to building sketches for other query columns, you can update src/storage/local_storage.cpp:LocalStorage::Append().

---

Happy coding! If you have any questions, please feel free to contact the TAs.
