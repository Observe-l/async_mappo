try:
    from onpolicy.envs.transformer_schedule.truck import Truck
except ModuleNotFoundError:
    from .truck import Truck

class Factory(object):
    def __init__(self, factory_id:str = 'Factory0',\
                 rate:float = 0.01,\
                 material:list = [None],\
                 product: str = 'P0') -> None:
        self.id = factory_id
        self.rate = rate
        self.product = product
        self.material = material
        self.reset()
    
    def reset(self) -> None:
        self.total_final_product = 0
        self.product_num = 0
        self.material_num = None
        if self.material[0] is not None:
            self.material_num = {m:0 for m in self.material}
    
    def produce(self) -> None:
        if self.material[0] is not None:
            if all(num >= self.rate for num in self.material_num.values()):
                self.product_num += self.rate
                self.total_final_product += self.rate
                for material in self.material:
                    self.material_num[material] -= self.rate
        else:
            self.product_num += self.rate

class Producer(object):
    '''
    product new product, load cargo, etc.
    '''
    def __init__(self, factory:dict, truck:list[Truck], transport_idx:dict, load_time:int = 600) -> None:
        self.factory = factory
        self.truck = truck
        self.transport_idx = transport_idx
        self.load_time = load_time
        self.final_product = ['A', 'B', 'C', 'D', 'E']
    
    def produce_step(self) -> None:
        # Produce all products
        for f in self.factory.values():
            f.produce()
        
        for t in self.truck:
            # Update the status of the truck
            t.truck_step()
            # Unload the goods from the truck
            if t.state == 'arrived':
                if t.product in self.factory[t.position].material:
                    self.factory[t.position].material_num[t.product] += t.weight
                    t.unload_goods(self.load_time)
                # If the trucks arrive the wrong factory, mark it as waiting
                else:
                    t.state = 'waiting'
            # Load the goods to the truck
            elif t.state == 'waiting' and t.weight == 0:
                if self.factory[t.position].product_num > t.capacity and self.factory[t.position].product not in self.final_product:
                    self.factory[t.position].product_num -= t.capacity
                    t.load_goods(self.factory[t.position].product, self.load_time)

                            