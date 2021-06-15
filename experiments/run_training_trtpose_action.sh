for py_file in (find ../src/s*.py)
do
    python $py_file
    echo [INFO] Finished running --- $py_file ---
done