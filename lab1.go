package main
import (
 "fmt"
 "gonum.org/v1/gonum/mat"
 "math"
 "math/rand"
)

func Random_matrix (n int,m int)(*mat.Dense){
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

func MatPrint(X mat.Matrix) {
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

func Make_diagonally_dominant(matrix mat.Matrix,col mat.Matrix)(*mat.Dense,*mat.Dense,*mat.Dense){
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

func Solve_monte_carlo(matrix mat.Matrix,col mat.Matrix)(*mat.Dense){
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

func Solve_gauss(matrix mat.Matrix,col mat.Matrix) (*mat.Dense){
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

func test_hardcoded(){
	matrix:= mat.NewDense (4,4 ,[] float64 {13, 12, 3, 16,0, 28, 17, 10,9, 1, 14, 3,10, 14, 3, 0})
	column:= mat.NewDense (4,1,[] float64 {242, 279, 111, 143})
	A,b,V := Make_diagonally_dominant(matrix, column)
	solution := mat.NewDense(4, 1, nil)
	solution.Product(V, Solve_monte_carlo(A, b))
	fmt.Println("Solving Ax = b:")
	fmt.Println("A:")
	MatPrint(matrix)	
	fmt.Println("b:")
	MatPrint(column)
	fmt.Println("solution")
	MatPrint(solution)
}

func main() {
	
	test_hardcoded()
}

