for py_file in ./s*.py
do
    python $py_file
    echo [INFO] Finished running --- $py_file ---
    echo ''
done
