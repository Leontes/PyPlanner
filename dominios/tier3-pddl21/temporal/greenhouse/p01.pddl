(define (problem greenhouse-1) (:domain greenhouse) (:objects t1 t2 - tank)
  (:init (raining) (sensor-on) (idle) (= (water t1) 0) (= (water t2) 0) (= (reserve) 0) (= (budget) 10))
  (:goal (and (logged) (>= (water t1) 6) (>= (water t2) 6) (>= (budget) 0))))
