# Choose the compiler and set compiler options

CPPCOMP  = g++
CPPOPTS  = -O3

FCOMP    = gfortran
FOPTS    = -O3

LDOPT =

# Set the list of programs to compile

PROGRAMS = test_chebyshev test_tensor test_odesolve test_kummer   \
           jacobi_quad laguerre_quad legendre_eval test_eq237_2


# Compile all of the test programs and the library

all	        : clean $(PROGRAMS) 

# List the dependencies for each module's test program


JACOBI_QUAD_FILES      = utils.o                       \
                         chebyshev.o                   \
                         odesolve.o                    \
                         kummer.o

LAGUERRE_QUAD_FILES    = utils.o                       \
                         chebyshev.o                   \
                         odesolve.o                    \
                         kummer.o

LEGENDRE_EVAL_FILES    = utils.o                       \
                         chebyshev.o                   \
                         tensor.o                      \
                         odesolve.o                    \
                         kummer.o                      \
                         kummer_legendre.o

KUMMER_FILES           = utils.o                       \
                         chebyshev.o                   \
                         odesolve.o                    \
                         kummer.o

ODESOLVE_FILES         = utils.o                       \
                         chebyshev.o                   \
                         odesolve.o

TENSOR_FILES           = utils.o                       \
                         chebyshev.o                   \
                         tensor.o

CHEBYSHEV_FILES        = utils.o                       \
                         chebyshev.o 

EQ237_FILES           =  utils.o                       \
                         chebyshev.o                   \
                         odesolve.o                    \
                         kummer.o



jacobi_quad.o          : $(JACOBI_QUAD_FILES) jacobi_quad.f90
jacobi_quad            : $(JACOBI_QUAD_FILES) jacobi_quad.o

laguerre_quad.o        : $(LAGUERRE_QUAD_FILES) laguerre_quad.f90
laguerre_quad          : $(LAGUERRE_QUAD_FILES) laguerre_quad.o

legendre_eval.o        : $(LEGENDRE_EVAL_FILES) legendre_eval.f90
legendre_eval          : $(LEGENDRE_EVAL_FILES) legendre_eval.o

test_kummer.o          : $(KUMMER_FILES) test_kummer.f90
test_kummer            : $(KUMMER_FILES) test_kummer.o

test_odesolve.o        : $(ODESOLVE_FILES) test_odesolve.f90
test_odesolve          : $(ODESOLVE_FILES) test_odesolve.o

test_tensor.o          : $(TENSOR_FILES) test_tensor.f90
test_tensor            : $(TENSOR_FILES) test_tensor.o

test_chebyshev.o       : $(CHEBYSHEV_FILES) test_chebyshev.f90
test_chebyshev         : $(CHEBYSHEV_FILES) test_chebyshev.o

test_eq237_2.o          : $(EQ237_FILES) test_eq237_2.f90
test_eq237_2            : $(EQ237_FILES) test_eq237_2.o




# Setup the general compilation rules

%		: %.o
	$(FCOMP) $(FOPTS) -o $@ $^ $(LDOPT)
#	@echo  
#	@echo 
#	@echo "---------[ $@     ]--------------------------------------------"
#	@echo 
#	@./$@
#	@echo 
#	@echo "--------------------------------------------------------------------------"
#	@echo 

%.o		: %.f90
	$(FCOMP) -c $(FOPTS)  $<

%.o		: %.f
	$(FCOMP) -c $(FOPTS)  $<

%.o		: %.cpp
	$(CPPCOMP) -c $(CPPOPTS)  $<

.PHONY: clean

clean:
	rm -f *.o *.mod *~ fort.*
	rm -f $(PROGRAMS)
	rm -f gn??? gn???.dat
