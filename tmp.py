from rod import *
from dbc import *
from potential import *
from geometry import *
from solver import *
import os
import warp as wp
import warp.sparse as sparse
import warp.optim.linear as linear

wp.init()
gravity = wp.constant(-1.0)

@wp.kernel
def initM(M: wp.array(dtype = float),
          J: wp.vec3, m: float, scale: float, n: int):
    i = wp.tid()
    if i <= 3 * n + 2:
        M[i] = m * scale
    else:
        j = i - (3 * n + 3)
        if j % 4 <= 2:
            M[i] = J[j % 4] * scale
        else:
            M[i] = 0.0

@wp.kernel
def initb(b: wp.array(dtype = float),
          M_star: wp.array(dtype = float),
          s: wp.array(dtype = float)):
    i = wp.tid()
    b[i] = M_star[i] * s[i]

@wp.kernel
def initGuessX(x: wp.array(dtype = float), 
               v: wp.array(dtype = float), 
               sx: wp.array(dtype = float),
               fExt: wp.array(dtype = float),
               dt: float, mInv: float):
    i = wp.tid()
    sx[i] = x[i] + dt * v[i] + dt * dt * fExt[i] * mInv
    if i % 3 == 2:
        sx[i] += dt * dt * gravity

@wp.kernel
def initOmega(w: wp.array(dtype = float), 
              sw: wp.array(dtype = float),
              tExt: wp.array(dtype = float),
              dt: float,
              J: wp.vec3,
              JInv: wp.vec3):
    i = wp.tid()
    base = 3 * i
    omega = wp.vec3(w[base], w[base + 1], w[base + 2])
    tau = wp.vec3(tExt[base], tExt[base + 1], tExt[base + 2])
    Jomega = wp.cw_mul(J, omega)
    res = omega + dt * wp.cw_mul(JInv, tau - wp.cross(omega, Jomega))
    sw[base] = res[0] 
    sw[base + 1] = res[1]
    sw[base + 2] = res[2]

@wp.kernel
def initGuessU(u: wp.array(dtype = float),
               sw: wp.array(dtype = float),
               su: wp.array(dtype = float),
               dt: float, n: int):
    i = wp.tid()
    base3, base4 = 3 * i, 4 * i
    swQuat = wp.quaternion(sw[base3], sw[base3 + 1], sw[base3 + 2], 0.0)
    uQuat = wp.quaternion(u[base4], u[base4 + 1], u[base4 + 2], u[base4 + 3])
    resQuat = uQuat + 0.5 * dt * (uQuat * swQuat)
    su[3 * n + 3 + base4] = resQuat[0]
    su[3 * n + 3 + base4 + 1] = resQuat[1]
    su[3 * n + 3 + base4 + 2] = resQuat[2]
    su[3 * n + 3 + base4 + 3] = resQuat[3]

@wp.kernel
def updateTrans(v: wp.array(dtype = float), 
                xLast: wp.array(dtype = float),
                xNext: wp.array(dtype = float),
                n: int, dtInv: float):
    i = wp.tid()
    v[i] = (xNext[i] - xLast[i]) * dtInv
    xLast[i] = xNext[i]

@wp.kernel
def updateRot(w: wp.array(dtype = float),
              uLast: wp.array(dtype = float),
              uNext: wp.array(dtype = float),
              n: int, dtInv: float):
    i = wp.tid()
    base3, base4, start = 3 * i, 4 * i, 3 * n + 3
    u1 = wp.quaternion(uNext[start + base4], uNext[start + base4 + 1], uNext[start + base4 + 2], uNext[start + base4 + 3])
    u0 = wp.quaternion(uLast[base4], uLast[base4 + 1], uLast[base4 + 2], uLast[base4 + 3])
    w1 = wp.quat_inverse(u0) * u1 * 2.0 * dtInv
    w[base3] = w1[0]
    w[base3 + 1] = w1[1]
    w[base3 + 2] = w1[2]
    uLast[base4] = uNext[start + base4]
    uLast[base4 + 1] = uNext[start + base4 + 1]
    uLast[base4 + 2] = uNext[start + base4 + 2]
    uLast[base4 + 3] = uNext[start + base4 + 3]

@wp.kernel
def normalizeQuaternion(q: wp.array(dtype = float),
                        n: int):
    i = wp.tid()
    base4 = 3 * n + 3 + 4 * i
    q0 = wp.quaternion(q[base4], q[base4 + 1], q[base4 + 2], q[base4 + 3])
    q1 = wp.normalize(q0)
    q[base4] = q1[0]
    q[base4 + 1] = q1[1]
    q[base4 + 2] = q1[2]
    q[base4 + 3] = q1[3]
    # print(q1)

@wp.kernel
def getQuaternion(result: wp.array(dtype = wp.quatf), 
                  q0: float, q1: float, q2: float, q3: float):
    result[0] = wp.quaternion(q0, q1, q2, q3)

class Sim:

    def __init__(self, fps: int = 30, stepPerFrame: int = 2, rod: Rod = Rod(), solverIter: int = 8):
        self.rod = rod
        self.n = rod.n

        self.JInv = wp.vec3(1.0 / rod.J1, 1.0 / rod.J2, 1.0 / rod.J3)
        self.mInv = 1.0 / self.rod.m
        self.lInv = 1 / rod.l
        self.wSE = rod.E * rod.A3 * rod.l
        self.wBT = 4 * rod.G * rod.J3 * self.lInv

        self.solverIter = solverIter
        self.dt = (1 / fps) / stepPerFrame
        self.dtInv = fps * stepPerFrame
        self.currentTime = 0
        
        self.transDBCx0 = []
        self.transDBCv = []
        self.transDBCindex = []
        self.rotDBCu0 = []
        self.rotDBCaxis = []
        self.rotDBCw = []
        self.rotDBCindex = []

        self.spheres = []
        # self.N = 1
        # while self.N < 7 * self.n + 3:
        #     self.N *= 2
        self.N = 7 * self.n + 3

        self.J = wp.vec3(rod.J1, rod.J2, rod.J3) * rod.lm
        self.M_star = wp.zeros(shape = (self.N, ), dtype = float)
        wp.launch(
            initM, dim = self.N,
            inputs = [self.M_star, self.J, self.rod.m, 1 / (self.dt * self.dt), self.n]
        )
        self.empty = wp.zeros(shape = (self.N, ), dtype = float)
        self.init_b = wp.zeros(shape = (self.N, ), dtype = float)
        self.b = wp.zeros_like(self.init_b)
    
        self.s = wp.zeros(shape = (self.N, ), dtype = float)
        self.sw = wp.zeros(shape = (3 * self.n + 3, ), dtype = float)
        self.q = wp.zeros_like(self.s)
        self.qNext = wp.zeros_like(self.s)
        self.collisionCount = wp.zeros(self.n + 1, dtype = int)

        self.r = wp.zeros(self.N, dtype = float)
        self.p = wp.zeros(self.N, dtype = float)
        self.x = wp.zeros(self.N, dtype = float)
        self.r_tilde = wp.zeros(self.N, dtype = float)
        self.Ap = wp.zeros(self.N, dtype = float)
        self.Ax = wp.zeros(self.N, dtype = float)
        self.temp = wp.zeros(self.N, dtype = float)

    def addTransDBC(self, nodeIndex: int, x0: wp.vec3, v: wp.vec3):
        self.transDBCx0 += [x0[0], x0[1], x0[2]]
        self.transDBCv += [v[0], v[1], v[2]]
        self.transDBCindex += [nodeIndex]

    def addRotDBC(self, edgeIndex: int, d0: wp.vec3, rotAxis: wp.vec3, w: float):
        u0 = getRotation(d0)
        self.rotDBCu0 += [u0[0], u0[1], u0[2], u0[3]]
        self.rotDBCaxis += [rotAxis]
        self.rotDBCw += [w]
        self.rotDBCindex += [edgeIndex]
    
    def addSphere(self, x0: wp.vec3, v: wp.vec3, radius: float):
        self.spheres += [initSphere(x0, v, radius)]

    def initBoundary(self):
        self.transDBCNum = len(self.transDBCindex)
        self.rotDBCNum = len(self.rotDBCindex)
        self.wp_transDBCx = wp.array(self.transDBCx0, dtype = float)
        self.wp_transDBCv = wp.array(self.transDBCv, dtype = float)
        self.wp_transDBCindex = wp.array(self.transDBCindex, dtype = int)
        
        self.wp_rotDBCu = wp.array(self.rotDBCu0, dtype = float)
        self.wp_rotDBCaxis = wp.array(self.rotDBCaxis, dtype = wp.vec3)
        self.wp_rotDBCw = wp.array(self.rotDBCw, dtype = float)
        self.wp_rotDBCindex = wp.array(self.rotDBCindex, dtype = int)

        self.sphereNum = len(self.spheres)
        self.wp_spheres = wp.array(self.spheres, dtype = Sphere)

    # def capture(self, fExt = wp.zeros(), tExt = wp.zeros()):


    def timeStep(self, fExt = wp.zeros(), tExt = wp.zeros()):
        self.currentTime += self.dt
        # update Dirichlet boundary conditions
        wp.launch(
            advanceTransDBC, dim = self.transDBCNum,
            inputs = [self.wp_transDBCx, self.wp_transDBCv, self.dt]
        )
        wp.launch(
            advanceRotDBC, self.rotDBCNum,
            inputs = [self.wp_rotDBCu, self.wp_rotDBCaxis, self.wp_rotDBCw, self.dt]
        )
        # # initialize guess
        # line 2
        wp.launch(
            initGuessX, dim = 3 * self.n + 3, 
            inputs = [self.rod.x, self.rod.v, self.s, fExt, self.dt, self.mInv]
        )
        # line 3
        wp.launch(
            initOmega, dim = self.n,
            inputs = [self.rod.w, self.sw, tExt, self.dt, self.J, self.JInv]
        )
        # line 4, 5
        wp.launch(
            initGuessU, dim = self.n, 
            inputs = [self.rod.u, self.sw, self.s, self.dt, self.n]
        )
        wp.copy(self.q, self.s)
        # iteration (line 7, 8, 9, 10)
        wp.launch(
            initb, dim = 3 * self.n + 3,
            inputs = [self.init_b, self.M_star, self.s]
        )
        for _ in range(self.solverIter):
            wp.copy(self.b, self.init_b)
            # accumulate all SE / BT potential
            wp.launch(
                potential, dim = (2, self.n),
                inputs = [self.q, self.b, self.lInv, self.n, self.wSE, self.wBT]
            )
            # accumulate all DBC potential
            wp.launch(
                DBCPotential, dim = self.transDBCNum + self.rotDBCNum,
                inputs = [
                    self.wp_transDBCx, self.wp_rotDBCu, self.q, 
                    self.wp_transDBCindex, self.wp_rotDBCindex, self.b, 
                    self.n, self.transDBCNum
                ]
            )
            self.collisionCount.fill_(0)
            # accumulate sphere collision potential
            wp.launch(
                sphereCollisionPotential, dim = (self.n + 1, self.sphereNum),
                inputs = [
                    self.q, self.wp_spheres, 
                    self.collisionCount, self.b, self.n, 
                    self.rod.radius, self.currentTime
                ]
            )
            # solve linear system
            wp.copy(self.qNext, self.s)
            self.conjugate_gradient()
            # normalize all quaternions
            wp.launch(
                normalizeQuaternion, self.n,
                inputs = [self.q, self.n]
            )
        # # fix dirichlet boundary conditions
        # wp.launch(
        #     DBCProject, dim = self.transDBCNum + self.rotDBCNum,
        #     inputs = [
        #         self.wp_transDBCx, self.wp_rotDBCu, q, 
        #         self.wp_transDBCindex, self.wp_rotDBCindex, self.n, self.transDBCNum
        #     ]
        # )
        # update velocity
        wp.launch(
            updateTrans, 3 * self.n + 3, 
            inputs = [self.rod.v, self.rod.x, self.q, self.n, self.dtInv]
        )
        # update angular velocity
        wp.launch(
            updateRot, self.n,
            inputs = [self.rod.w, self.rod.u, self.q, self.n, self.dtInv]   
        )

    def conjugate_gradient(self, maxIter = 25, tol = 0):
        BLOCK = 2 ** 5
        NUM_BLOCK = (self.N - 1) // BLOCK + 1;
        wp.copy(self.x, self.qNext)
        diagonal = wp.zeros((self.N, ), dtype = float)
        wp.launch(
            queryDiagonal, dim = (5, self.N), 
            inputs = [
                self.wp_transDBCindex, self.wp_rotDBCindex, self.collisionCount,
                self.M_star, diagonal,
                self.n, self.transDBCNum, self.rotDBCNum,
                self.wSE, self.wBT, self.lInv
            ]
        )
        wp.launch(
            precondition, dim = self.N,
            inputs = [diagonal, self.b]
        )
        self.Ax.fill_(0.0)
        wp.launch(
            queryProduct, dim = (5, self.N), 
            inputs = [
                self.wp_transDBCindex, self.wp_rotDBCindex, self.collisionCount,
                self.M_star, self.x, self.Ax, 
                self.n, self.transDBCNum, self.rotDBCNum,
                self.wSE, self.wBT, self.lInv
            ]
        )
        wp.launch(
            precondition, dim = self.N,
            inputs = [diagonal, self.Ax]
        )
        wp.launch(
            vectorAdd, dim = self.N,
            inputs = [
                self.b, self.Ax, self.r, 
                wp.array([1.0], dtype = float), wp.array([1.0], dtype = float), -1.0
            ]
        )
        wp.copy(self.p, self.r)
        r_square = wp.zeros(1, dtype = float)
        r_tilde_square = wp.zeros(1, dtype = float)
        pap = wp.zeros(1, dtype = float)
        wp.launch(
            dot, dim = NUM_BLOCK,
            inputs = [self.r, self.r, r_square, NUM_BLOCK, self.N]
        )
        wp.capture_begin(device = "cuda:0")
        self.Ap.fill_(0.0)
        wp.launch(
            queryProduct, dim = (5, self.N), 
            inputs = [
                self.wp_transDBCindex, self.wp_rotDBCindex, self.collisionCount,
                self.M_star, self.p, self.Ap, 
                self.n, self.transDBCNum, self.rotDBCNum,
                self.wSE, self.wBT, self.lInv
            ]
        )
        wp.launch(
            precondition, dim = self.N,
            inputs = [diagonal, self.Ap]
        )
        pap.fill_(0.0)
        wp.launch(
            dot, dim = NUM_BLOCK,
            inputs = [self.p, self.Ap, pap, NUM_BLOCK, self.N]
        )
        wp.launch(
            vectorAdd, dim = self.N,
            inputs = [self.x, self.p, self.x, r_square, pap, 1.0]
        )
        wp.copy(self.r_tilde, self.r)
        wp.copy(r_tilde_square, r_square)
        wp.launch(
            vectorAdd, dim = self.N,
            inputs = [self.r, self.Ap, self.r, r_square, pap, -1.0]
        )
        r_square.fill_(0.0)
        wp.launch(
            dot, dim = NUM_BLOCK,
            inputs = [self.r, self.r, r_square, NUM_BLOCK, self.N]
        )
        wp.launch(
            vectorAdd, dim = self.N,
            inputs = [self.r, self.p, self.p, r_square, r_tilde_square, 1.0]
        )
        graph = wp.capture_end(device = "cuda:0")
        for _ in range(maxIter):
            wp.capture_launch(graph)
        wp.copy(self.q, self.x)

    def export(self, fileName, filePath, index, resolution):
        vertices = []
        nodeIndex = []
        edgeIndex = []
        faces = []
        
        def addVertex(theta, trans, orient, vertices):
            a, b = 2, 0.2
            initVertex = wp.vec3(wp.cos(theta) * self.rod.radius * a, wp.sin(theta) * self.rod.radius * b, 0)
            initVertex = wp.quat_rotate(orient, initVertex)
            vertex = initVertex + trans
            vertices += [vertex]

        def createQuaternion(q0, q1, q2, q3):
            buffer = wp.zeros(1, dtype = wp.quatf)
            wp.launch(getQuaternion, 1, inputs = [buffer, q0, q1, q2, q3])
            return buffer.numpy()[0]
        
        x = self.rod.x.numpy()
        u = self.rod.u.numpy()
        for i in range(self.n + 1): # vertices at node
            base3, base4 = 3 * i, 4 * i
            trans = wp.vec3(x[base3], x[base3 + 1], x[base3 + 2])
            if i == 0:
                orient = createQuaternion(u[base4], u[base4 + 1], u[base4 + 2], u[base4 + 3])
            elif i == self.n:
                orient = createQuaternion(u[base4 - 4], u[base4 - 3], u[base4 - 2], u[base4 - 1])
            else:
                lastOrient = createQuaternion(u[base4 - 4], u[base4 - 3], u[base4 - 2], u[base4 - 1])
                nextOrient = createQuaternion(u[base4], u[base4 + 1], u[base4 + 2], u[base4 + 3])
                orient = wp.quat_slerp(lastOrient, nextOrient, 0.5)
            nodeIndex += [len(vertices)]
            for j in range(resolution):
                theta = 2 * wp.PI * j / resolution
                addVertex(theta, trans, orient, vertices)
        
        for i in range(self.n): # vertices at edge
            base3, base4 = 3 * i, 4 * i
            lastNode = wp.vec3(x[base3], x[base3 + 1], x[base3 + 2])
            nextNode = wp.vec3(x[base3 + 3], x[base3 + 4], x[base3 + 5])
            trans = 0.5 * (lastNode + nextNode)
            orient = createQuaternion(u[base4], u[base4 + 1], u[base4 + 2], u[base4 + 3])
            edgeIndex += [len(vertices)]
            for j in range(resolution):
                theta = 2 * wp.PI * j / resolution
                addVertex(theta, trans, orient, vertices)
        
        vertices += [wp.vec3(x[0], x[1], x[2]),
                     wp.vec3(x[3 * self.n], x[3 * self.n + 1], x[3 * self.n + 2])]
        end0 = len(vertices) - 2
        end1 = end0 + 1

        for i in range(self.n):
            for j in range(resolution):
                jNext = j + 1 if j < resolution - 1 else 0
                jLast = j - 1 if j > 0 else resolution - 1
                # faces on first half of segements
                faces += [[nodeIndex[i] + j, edgeIndex[i] + j, edgeIndex[i] + jNext]]
                faces += [[nodeIndex[i] + j, nodeIndex[i] + jLast, edgeIndex[i] + j]]
                # faces on second half of segements
                faces += [[nodeIndex[i + 1] + j, edgeIndex[i] + j, edgeIndex[i] + jLast]]
                faces += [[nodeIndex[i + 1] + j, nodeIndex[i + 1] + jNext, edgeIndex[i] + j]]
        
        for j in range(resolution):
            jNext = j + 1 if j < resolution - 1 else 0
            # faces on first end
            faces += [[nodeIndex[0] + j, nodeIndex[0] + jNext, end0]]
            # faces on second end
            faces += [[nodeIndex[self.n] + j, nodeIndex[self.n] + jNext, end1]]
        
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), filePath)
        with open(os.path.join(path, fileName) + "_{}.obj".format(index), mode = 'w') as file:
            for x in vertices:
                file.write("v {:.6f} {:.6f} {:.6f}\n".format(x[0], x[1], x[2]))
            for f in faces:
                file.write("f {} {} {}\n".format(f[0] + 1, f[1] + 1, f[2] + 1))
        
        # write spheres
        with open(os.path.join(path, fileName) + "_sphere_{}.x3d".format(index), mode = 'w') as file:
            file.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?> \n\
    <X3D profile=\"Interchange\" version=\"3.3\" xmlns:xsd=\"http://www.w3.org/2001/XMLSchema-instance\" xsd:noNamespaceSchemaLocation=\"http://www.web3d.org/specifications/x3d-3.3.xsd\"> \n\
        <Scene>"
            )
            
            for i in range(self.sphereNum):
                sphere = self.spheres[i]
                center = sphere.pos + self.currentTime * sphere.velocity
                file.write("\n\
            <Transform translation=\"{:.6f} {:.6f} {:.6f}\"> \n\
                <Shape> \n\
                    <Sphere radius=\"{:.6f}\"/> \n\
                </Shape> \n\
            </Transform>".format(center[0], center[1], center[2], sphere.radius)           
                )
            
            file.write("\n\
        </Scene> \n\
    </X3D>"
            )
            
            