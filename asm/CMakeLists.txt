add_library(ke_grad_ksimd ke_grad.cpp)
target_link_libraries(ke_grad_ksimd omega_split_ksimd)
target_compile_options(ke_grad_ksimd PRIVATE -save-temps)

add_library(ke_grad_osimd ke_grad.cpp)
target_link_libraries(ke_grad_osimd omega_split_osimd)
target_compile_options(ke_grad_osimd PRIVATE -save-temps)
