package main
import(
 "fmt"
 "gonum.org/v1/gonum/mat"
 "math"
 "time"
 "testing"
)
func Test_Monte_Carlo(t *testing.T){
	max_dim := 50
	epsilon := 1e-5

	total_monte:=0.0
	total_gauss:=0.0
	for i:=3;i<max_dim;i++{
		matrix:=Random_matrix(i,i)
		column:=Random_matrix(i,1)

		start:=time.Now()
		A,b,V:=Make_diagonally_dominant(matrix,column)
		if !is_diagonally_dominant(A){
			fmt.Printf("Error! {%v}x{%v} matrix is not diagonally dominant\n",i, i)
			continue
		}
		time_diagonal:=time.Since(start)
		
		start=time.Now()
		res_monte:=mat.NewDense(i,1,nil)
		res_monte.Product(V,Solve_monte_carlo(A,b))
		time_monte:=time.Since(start)
		
		start=time.Now()
		res_gauss:=Solve_gauss(matrix,column)
		time_gauss:=time.Since(start)

		for j:=0;j<i;j++{
			if (math.Abs(res_gauss.At(j, 0)-res_monte.At(j, 0))>=epsilon){
				fmt.Println("------------------------------------")
				t.Errorf("Test fails for random matrix(%vx%v)\n",i, i)
				fmt.Println("Monte carlo result : ")
				MatPrint(res_monte)
				fmt.Println("Gauss result : ")
				MatPrint(res_gauss)
			  	return		
			}
		}
		total_monte+=float64(time_monte)
		total_gauss+=float64(time_gauss)
		fmt.Printf("Passed for random matrix %vx%v. Time: Diagonalization - %v, Monte Carlo - %v, Gauss - %v\n",
              i, i, time_diagonal, time_monte, time_gauss)
	}

}


