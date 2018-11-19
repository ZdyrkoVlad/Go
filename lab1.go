package main

import (
 "fmt"
 "gonum.org/v1/gonum/mat"
 "math"
 "math/rand"
 "time"
)
// var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to `file`")

func random_matrix (n int,m int)(*mat.Dense){
	s:=mat.NewDense(n,m,nil)
	for i:=0;i<n;i++{
		for j:=0;j<m;j++{
			s.Set(i,j,float64(rand.Intn(100)))
		}
	}
	return s
}


func make_identity(w int)(*mat.Dense){
	tity:=mat.NewDense(w,w,nil)
	for i:=0;i<w;i++{
		tity.Set(i,i,1)
	}
	return tity
}

func matPrint(X mat.Matrix) {
 fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
 fmt.Printf("%v\n", fa)
}

func svd_2x2(matrix mat.Matrix)(*mat.Dense,*mat.Dense,*mat.Dense){
	e:=(matrix.At(0,0)+matrix.At(1,1))/2
	f:=(matrix.At(0,0)-matrix.At(1,1))/2
	g:=(matrix.At(1,0)+matrix.At(0,1))/2
	h:=(matrix.At(1,0)-matrix.At(0,1))/2
	a1:=math.Atan2(g,f)
	a2:=math.Atan2(h,e)
	alpha:=(a1+a2)/2
	beta:=(a2-a1)/2
	q:=math.Sqrt(e * e + h * h)
	r:=math.Sqrt(f*f+g*g)
	u:=mat.NewDense(2,2,[] float64 {math.Cos(alpha),-math.Sin(alpha),math.Sin(alpha),math.Cos(alpha)})
	vt:=mat.NewDense(2,2,[] float64{math.Cos(beta),-math.Sin(beta),math.Sin(beta),math.Cos(beta)})
	s:=mat.NewDense(2,2,[] float64{q+r,0,0,q-r})
	return u,s,vt
}

func matrix_norm(matrix mat.Matrix)(float64){
	max_sum:=0.0
	w,_:=matrix.Dims()
	for i:=0;i<w;i++{
		s:=0.0
		for j:=0;j<w;j++{
			s=s+math.Abs(matrix.At(i,j))
		}
		max_sum=math.Max(max_sum,s)
	}
	return max_sum
} 
func max_element_position(matrix mat.Matrix)(int,int){
	pos_i:=0
	pos_j:=0
	max_element:=0.0
	w,_:=matrix.Dims()
	for i:=0;i<w;i++{
		for j:=0;j<w;j++{
			if i!=j && max_element<math.Abs(matrix.At(i,j)){
				max_element=math.Abs(matrix.At(i,j))
				pos_i=i
				pos_j=j
			}
		}
	}
	return pos_i,pos_j	
}

func is_diagonally_dominant(matrix mat.Matrix)(bool){
	w,_:=matrix.Dims()
	for i:=0;i<w;i++{
		sum:=0.0
		for j:=0;j<w;j++{
			if i!=j{
				sum+=math.Abs(matrix.At(i,j))
			}			
		}
		if sum>=math.Abs(matrix.At(i,i)){
			return false
		}

	}
	return true
}

func make_diagonally_dominant(matrix mat.Matrix,col mat.Matrix)(*mat.Dense,*mat.Dense,*mat.Dense){
	A:=mat.DenseCopyOf(matrix)
	b:=mat.DenseCopyOf(col)
	w,_:=matrix.Dims()
	V:=make_identity(w)
	m:=w*w*2
	for i:=0; i < m; i++{
		max_i,max_j:=max_element_position(A)
		pos_i:=int(math.Min(float64(max_i),float64(max_j)))
		pos_j:=int(math.Max(float64(max_i),float64(max_j)))
		Q:=mat.NewDense(2,2,[] float64{A.At(pos_i,pos_i),A.At(pos_i,pos_j),A.At(pos_j,pos_i),A.At(pos_j,pos_j)})
		u,_,vt:=svd_2x2(Q)
		Uk:=make_identity(w)
		Uk.Set(pos_i,pos_i,u.At(0,0))
		Uk.Set(pos_i,pos_j,u.At(1,0))
		Uk.Set(pos_j,pos_i,u.At(0,1))
		Uk.Set(pos_j,pos_j,u.At(1,1))
		Vk:=make_identity(w)
		Vk.Set(pos_i,pos_i,vt.At(0,0))
		Vk.Set(pos_i,pos_j,vt.At(1,0))
		Vk.Set(pos_j,pos_i,vt.At(0,1))
		Vk.Set(pos_j,pos_j,vt.At(1,1))
		A.Product(Uk,A,Vk)
		b.Product(Uk,b)
		V.Product(V,Vk)
	}
	return A,b,V
}



func seidel_condition(x mat.Matrix,prev_x mat.Matrix,q float64,eps float64)(bool){
	diff:=0.0
	w,_:=x.Dims()
	for i:=0;i<w;i++{
		diff+=(x.At(i,0)-prev_x.At(i,0))*(x.At(i,0)-prev_x.At(i,0))
	}
	if math.Sqrt(diff)<=math.Abs(q*eps/(1-q)){
		return false
	}
	return true
}

func solve_monte_carlo(matrix mat.Matrix,col mat.Matrix)(*mat.Dense){
	A:=mat.DenseCopyOf(matrix)
	b:=mat.DenseCopyOf(col)
	w,_:=A.Dims()


	for i:=0;i<w;i++{
		diag:=A.At(i,i)
		for j:=0;j<w;j++{
			A.Set(i,j,A.At(i,j)/-diag)
		}
		b.Set(i,0,b.At(i,0)/diag)
		A.Set(i,i,0)

	}

	P:=mat.DenseCopyOf(A)
	for i:=0;i<w;i++{
		row_sum:=0.0
		for j:=0;j<w;j++{
			row_sum+=math.Abs(P.At(i,j))
		}
		for j:=0;j<w;j++{
			P.Set(i,j,math.Abs(P.At(i,j))/row_sum)
		}	
	}

	M:=5
	N:=10
	final_x:=mat.NewDense(w,1,nil)
	for s:=0;s<N;s++{
		x:=mat.DenseCopyOf(b)
		for t:=0;t<M;t++{
			prev_x:=mat.DenseCopyOf(x)
			for i:=0; i<w;i++{
				r_num:=rand.Float64()
				random_ind:=0
				for r_num>=0.0 && random_ind<w{
					r_num-=P.At(i,random_ind)
					random_ind+=1
				}
				random_ind-=1
				if random_ind<i{
					x.Set(i,0,b.At(i,0)+A.At(i,random_ind)*x.At(random_ind,0)/P.At(i,random_ind))
				}else{
					x.Set(i,0,b.At(i,0)+A.At(i,random_ind)*prev_x.At(random_ind,0)/P.At(i,random_ind))
				}

			}

		}
		final_x.Add(final_x,x)
	}
	final_x.Scale(1.0/float64(N),final_x)
	return final_x
	
}

func solve_gauss(matrix mat.Matrix,col mat.Matrix) (*mat.Dense){
	A:= mat.DenseCopyOf(matrix)
	b:= mat.DenseCopyOf(col)
	w,h:=A.Dims()

	for i:=0;i<w;i++{
		diag:=A.At(i,i)
		for j:=0;j<h;j++{
			A.Set(i,j,(A.At(i,j)/diag))	
		}
		b.Set(i,0,(b.At(i,0)/diag))
		
		for j:=0;j<w;j++{
			if j!=i{
				elem:=A.At(j,i)
				for k:=0;k<w;k++{
					A.Set(j,k,A.At(j,k)-(A.At(i,k)*elem))			
				}
				b.Set(j,0,b.At(j,0)-(b.At(i,0)*elem))	
			}
			
		}
		

		
	}
	return b
}

func test_all(max_dim int ,epsilon float64){
	total_monte:=0.0
	total_gauss:=0.0
	for i:=3;i<max_dim;i++{
		matrix:=random_matrix(i,i)
		column:=random_matrix(i,1)

		start:=time.Now()
		A,b,V:=make_diagonally_dominant(matrix,column)
		if !is_diagonally_dominant(A){
			fmt.Printf("Error! {%v}x{%v} matrix is not diagonally dominant\n",i, i)
			continue
		}
		time_diagonal:=time.Since(start)
		
		start=time.Now()
		res_monte:=mat.NewDense(i,1,nil)
		res_monte.Product(V,solve_monte_carlo(A,b))
		time_monte:=time.Since(start)
		
		start=time.Now()
		res_gauss:=solve_gauss(matrix,column)
		time_gauss:=time.Since(start)

		for j:=0;j<i;j++{
			if (math.Abs(res_gauss.At(j, 0)-res_monte.At(j, 0))>=epsilon){
				fmt.Println("------------------------------------")
				fmt.Printf("Test fails for random matrix(%vx%v)\n",i, i)
				fmt.Println("Monte carlo result : ")
				matPrint(res_monte)
				fmt.Println("Gauss result : ")
				matPrint(res_gauss)
			  	return		
			}
		}
		total_monte+=float64(time_monte)
		total_gauss+=float64(time_gauss)
		fmt.Printf("Passed for random matrix %vx%v. Time: Diagonalization - %v, Monte Carlo - %v, Gauss - %v\n",
              i, i, time_diagonal, time_monte, time_gauss)
	}

}

func test_hardcoded(){
	matrix:= mat.NewDense (4,4 ,[] float64 {13, 12, 3, 16,0, 28, 17, 10,9, 1, 14, 3,10, 14, 3, 0})
	column:= mat.NewDense (4,1,[] float64 {242, 279, 111, 143})
	A,b,V := make_diagonally_dominant(matrix, column)
	solution := mat.NewDense(4, 1, nil)
	solution.Product(V, solve_monte_carlo(A, b))
	fmt.Println("Solving Ax = b:")
	fmt.Println("A:")
	matPrint(matrix)	
	fmt.Println("b:")
	matPrint(column)
	fmt.Println("solution")
	matPrint(solution)
}


func main() {
	// if *cpuprofile != "" {
	//         f, err := os.Create(*cpuprofile)
	//         if err != nil {
	//             log.Fatal("could not create CPU profile: ", err)
	//         }
	//         if err := pprof.StartCPUProfile(f); err != nil {
	//             log.Fatal("could not start CPU profile: ", err)
	//         }
	//         defer pprof.StopCPUProfile()
	//     }

	test_all(100, 1e-5)
}
