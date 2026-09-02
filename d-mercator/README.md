Ejecutar create_graph_file_dmercatgor.ipynb

ir a: self_similarity/d-mercator
sudo python3 run_dmercator_docker.py -i ../main_branch/github/self-similarity-in-social-movements/d-mercator/graphs/ch/394693/394693.edge -d 1 -v ../main_branch/github/self-similarity-in-social-movements/d-mercator/graphs/ch/394693/394693.edge
y esperamos (comporbar archivo inf_log hasta que acabe, que como es docker está en segundo plano) 

Acaba cuando inf_log imprime el "==========================================================================================="

Una vez que está generado, si queremos generar el pdf, hacemos (source main_branch/github/self-similarity-in-social-movements/):  python3 pdf_d-mercator.py graphs/ch/394693/394693.edge (el titulo del pdf esta hardcodeado!)