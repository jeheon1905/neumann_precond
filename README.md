1. run_files.py

    이 코드는 cif 랑 sdf 파일이 존재하는 경로와 계산 material 을 넣으면 일치하는 파일을 찾아 시스템 구축 후 계산을 진행하는 코드이다. 
    cif, sdf 파일의 이름은 {material}_sdf, {material}_cif 로 존재하여야 한다.
    파일 내부에 존재하는 여러 인자들을 주면, 그에 맞는 시스템을 구축하여 계산을 수행한다. 

    --phase 옵션은 phase = "scf" 이면, 일반적인 scf 계산을 수행, phase = "fixed" 이면 밀도 파일을 사용한다. 또한 --density_filename 은 scf 이후 나타난 밀도 파일을 저장하는 파일 이름을 설정 하는 것이다.

        phase = "scf", density_filename = None  이면 density 파일을 저장하지 않고 scf 과정을 수행
        phase = "scf", density_filename = "results" 이면  density 파일을 results 에 저장
        phase - "fixed", density_filename = "results" 이면 scf 에서 구했던 density 파일을이용하는 것  

    ex) python run_files.py --material diamond --precond shift-and-invert --spacing 0.2 --supercell 3 3 3 --dir "./". --phase "scf" --density_filename "density_diamond", 


2. calculation_files.py
    이 코드는 특정 시스템 별로 조건을 달리하여 계산하고 싶을 때 조건을 설정하고 반복적인 방법으로 구하는 것이다. 이 코드는 반복적으로 run_files.py 를 실행하는 코드이다. 이후 결과 계산 파일은 total_diagonalizatio time 순으로 정렬되어 디랙토리로 모이게 된다.,

    ex) python calculation_files.py  처럼 args 는 없고 파일 내부에서 설정하도록 코드 작성 
