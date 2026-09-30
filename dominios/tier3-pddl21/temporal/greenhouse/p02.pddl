(define (problem greenhouse-2) (:domain greenhouse) (:objects t1 t2 - tank)
  (:init (sensor-on) (idle) (= (water t1) 0) (= (water t2) 0) (= (reserve) 0) (= (budget) 10))
  (:goal (and (logged) (>= (water t1) 3) (>= (water t2) 3))))
