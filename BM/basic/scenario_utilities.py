from dataclasses import dataclass, replace
from os import name
from smarts.sstudio import types as t


@dataclass
class Target:
    road_segment_name: str
    lane: int
    offset_on_lane: float

def merge_traffic(s: t.Traffic, o: t.Traffic):
    return t.Traffic([*s.flows, *o.flows])

def merge(s: t.Scenario, o: t.Scenario) -> t.Scenario:
    
    s_t:dict =s.traffic.copy() or {}
    o_t=o.traffic or {}

    s_sam=s.social_agent_missions.copy() or {}
    o_sam=o.social_agent_missions or {}

    s_em=s.ego_missions or []
    o_em=o.ego_missions or []

    s_b=s.bubbles or []
    o_b=o.bubbles or []

    s_fm=s.friction_maps or []
    o_fm=o.friction_maps or []

    s_th=s.traffic_histories or []
    o_th=o.traffic_histories or []

    keys={s_t.keys()} | {o_t.keys()}
    traffic={}
    for key in keys:
        if key in o_t and key in s_t:
            traffic[key] = merge_traffic(s_t[key], o_t[key])

    scenario = t.Scenario(
        traffic=traffic,
        ego_missions=[*s_em, *o_em],
        social_agent_missions=s_sam.update(o_sam),
        bubbles=[*s_b, *o_b],
        friction_maps=[*s_fm, *o_fm],
        traffic_histories=[*s_th, *o_th]
    )
    return scenario

class ScenarioGenerator:
    def __init__(self, target: Target, default_actor = None) -> None:
        self._commands = []
        self._scenario = t.Scenario()
        self._target: Target = target
        self._actor_id: int = 0
        self._default_traffic_actor = default_actor or t.TrafficActor(
            f"d", speed=t.Distribution(mean=1, sigma=0)
        )

    def update_default_vehicle(self, **kwargs):
        self._default_traffic_actor = replace(self._default_traffic_actor, kwargs)


    def vehicle_on_lane(self, offset_m, name=None):
        def add_vehicle():
            ct: Target=self._target
            traffic = t.Traffic(
                flows=[
                    t.Flow(
                        route=t.Route(
                            begin=(ct.road_segment_name, ct.lane, ct.offset_on_lane + offset_m),
                            end=(ct.road_segment_name, ct.lane, "max"),
                        ),
                        rate=1,
                        actors={t.TrafficActor(name or f"t_{self._actor_id}"): 1},
                    ),
                ]
            )
            scenario = t.Scenario(traffic=traffic)
            self._scenario = merge(self._scenario, scenario)
            self._actor_id += 1
        
        self._commands.append(add_vehicle)

    def relative_vehicle(self, offset_m, relative_lane, name=None):
        assert self._target.lane + relative_lane > 0
        def add_vehicle():
            ct: Target=self._target
            traffic = t.Traffic(
                flows=[
                    t.Flow(
                        route=t.Route(
                            begin=(ct.road_segment_name, ct.lane, ct.offset_on_lane + offset_m),
                            end=(ct.road_segment_name, ct.lane, "max"),
                        ),
                        rate=1,
                        actors={t.TrafficActor(name or f"t_{self._actor_id}"): 1},
                    ),
                ]
            )

            scenario = t.Scenario(traffic=traffic)
            self._scenario = merge(self._scenario, scenario)
            self._actor_id += 1
        
        self._commands.append(add_vehicle)

    def parked_vehicle(self, offset_m, lane=0, name=None):
        assert lane >= 0
        def add_parked_vehicle():
            ct: Target=self._target
            traffic = t.Traffic(
                flows=[
                    t.Flow(
                        route=t.Route(
                            begin=(ct.road_segment_name, lane, ct.offset_on_lane + offset_m),
                            end=(ct.road_segment_name, lane, "max"),
                        ),
                        rate=1,
                        actors={
                            replace(self._default_traffic_actor, name=name or f"t_{self._actor_id+1}", depart_speed=0, speed=t.t.Distribution(mean=0, sigma=0)): 1
                        },
                    ),
                ]
            )
            scenario = t.Scenario(traffic=traffic)
            self._scenario = merge(self._scenario, scenario)
            self._actor_id += 1
        
        self._commands.append(add_parked_vehicle)

    def boxed_in(self, front, back, speed_m=None):
        def add_box_in():
            ct: Target=self._target
            flows=[
                t.Flow(
                    route=t.Route(
                        begin=(ct.road_segment_name, ct.lane, ct.offset_on_lane + front),
                        end=(ct.road_segment_name, ct.lane, "max"),
                    ),
                    rate=1,
                    actors={replace(self._default_traffic_actor, name=f"t_{self._actor_id+1}"): 1},
                ),
                t.Flow(
                    route=t.Route(
                        begin=(ct.road_segment_name, ct.lane, ct.offset_on_lane + back),
                        end=(ct.road_segment_name, ct.lane, "max"),
                    ),
                    rate=1,
                    actors={replace(self._default_traffic_actor, name=f"t_{self._actor_id+1}"): 1},
                ),
            ]
            if ct.lane > 0:
                flows.append(
                    t.Flow(
                        route=t.Route(
                            begin=(ct.road_segment_name, ct.lane-1, ct.offset_on_lane),
                            end=(ct.road_segment_name, ct.lane-1, "max"),
                        ),
                        rate=1,
                        actors={replace(self._default_traffic_actor, name=f"t_{self._actor_id+1}"): 1},
                    )
                )
            self._actor_id += 4
            traffic = t.Traffic(flows=flows)
            scenario = t.Scenario(traffic=traffic)
            self._scenario = merge(self._scenario, scenario)

        self._commands.append(add_box_in)

    def add_mission(self, mission):
        def add_mission():
            missions = self._scenario.ego_missions or []
            self._scenario = replace(self._scenario, ego_missions=[*missions, mission])

        self._commands.append(add_mission)

    def add_dummy_target(self, name):
        def add_dummy():
            ct: Target=self._target
            traffic = t.Traffic(
                flows=[
                    t.Flow(
                        route=t.Route(
                            begin=(ct.road_segment_name, ct.lane, ct.offset_on_lane),
                            end=(ct.road_segment_name, ct.lane, "max"),
                        ),
                        rate=1,
                        actors={
                            replace(self._default_traffic_actor, name=name or f"t_{self._actor_id+1}", depart_speed=0, speed=t.t.Distribution(mean=0, sigma=0)): 1
                        },
                    ),
                ]
            )            
            scenario = t.Scenario(traffic=traffic)
            self._scenario = merge(self._scenario, scenario)
            self._actor_id += 1

        self._commands.append(add_dummy)
    def merge(self, scenario):
        def merge_scenarios():
            self._scenario = merge(self._scenario, scenario)

        self._commands.append(merge_scenarios)

    def resolve(self):

        for c in self._commands:
            c()

        return self._scenario